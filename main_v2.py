# %%
import logging
import multiprocessing
import os

# import aiofiles
import pathlib
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from nccl_comm import *
from nccl_primitives import *
from gpu import *
import argparse
import time

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# %%
def construct_communicators(
    comm_info: pd.DataFrame, comm_ring_info: pd.DataFrame, comm_tree_info: pd.DataFrame
) -> Tuple[Dict[str, Communicator], Dict[Tuple[str, int], GPUDevice]]:
    logger.info("constructing communicator objects")

    # construct GPUDevice objects
    gpus_df = comm_info[["nodeId", "pid"]].drop_duplicates()
    gpu_devices = {
        (row["nodeId"], row["pid"]): GPUDevice(rank=i, node_id=row["nodeId"], pid=row["pid"])
        for i, (_, row) in enumerate(gpus_df.iterrows())
    }
    comm_info = comm_info.copy()
    # Store gpu_id (nodeId, pid) tuples for Communicator construction
    comm_info["gpu_id"] = comm_info.apply(
        lambda row: (row["nodeId"], row["pid"]), axis=1
    )
    # Also keep reference to GPUDevice for later use
    comm_info["gpu"] = comm_info.apply(
        lambda row: gpu_devices[(row["nodeId"], row["pid"])], axis=1
    )
    comm_gpus_df = (
        comm_info.sort_values("rank")
        .groupby(["commId"])
        .aggregate({"gpu_id": list})
        .reset_index()
    )
    communicators = {
        row["commId"]: Communicator(row["commId"], row["gpu_id"])
        for _, row in comm_gpus_df.iterrows()
    }

    comm_ring_info = comm_ring_info.sort_values("channelId")
    for _, row in comm_ring_info.iterrows():
        communicators[row["commId"]].add_ring_topo(
            row["myRank"], row["prevRank"], row["nextRank"]
        )

    comm_tree_info = comm_tree_info.sort_values("channelId")
    for _, row in comm_tree_info.iterrows():
        children = [
            c
            for c in [row["child1Rank"], row["child2Rank"], row["child3Rank"]]
            if c >= 0
        ]
        communicators[row["commId"]].add_tree_topo(
            row["myRank"], row["parentRank"], children
        )

    return communicators, gpu_devices


algo_mapping = {
    0: CollAlgo.TREE,
    1: CollAlgo.RING,
}
proto_mapping = {
    0: NCCLProto.LL,
    1: NCCLProto.LL128,
    2: NCCLProto.SIMPLE,
}

context_labels = {"Other": 0, "PP": 1, "DP": 2}


# Global variables for worker processes (set via Pool initializer for copy-on-write efficiency)
_WORKER_GPU_DATA = None
_WORKER_GPU_ID2GOAL_RANK = None
_WORKER_OUTPUT_DIR = None
_WORKER_MERGED_STREAMS = None


def _init_worker(gpu_data_dict, gpu_id2goal_rank, output_dir, merged_streams):
    """
    Initialize worker process with shared data.
    This is called once per worker at fork time, enabling copy-on-write memory sharing.
    """
    global _WORKER_GPU_DATA, _WORKER_GPU_ID2GOAL_RANK, _WORKER_OUTPUT_DIR, _WORKER_MERGED_STREAMS
    _WORKER_GPU_DATA = gpu_data_dict
    _WORKER_GPU_ID2GOAL_RANK = gpu_id2goal_rank
    _WORKER_OUTPUT_DIR = output_dir
    _WORKER_MERGED_STREAMS = merged_streams


def init_and_generate_goal_for_gpu(args_tuple):
    """
    Worker function for parallel GPU initialization and goal generation.
    Uses global data set via initializer (copy-on-write friendly).
    Only receives small arguments via pickle.
    
    Args:
        args_tuple: (gpu_id, rank, nic, init_counter, gen_counter, counter_lock)
    
    init_counter, gen_counter: multiprocessing.Value counters for progress tracking
    counter_lock: multiprocessing.Lock to protect counter updates
    """
    gpu_id, rank, nic, init_counter, gen_counter, counter_lock = args_tuple
    
    # Access data from globals (copy-on-write, no pickle overhead)
    gpu_data = _WORKER_GPU_DATA[gpu_id]
    gpu_id2goal_rank = _WORKER_GPU_ID2GOAL_RANK
    output_dir = _WORKER_OUTPUT_DIR
    merged_streams = _WORKER_MERGED_STREAMS
    
    # Reconstruct GPUDevice in this process
    gpu = GPUDevice(rank=rank, node_id=gpu_data["node_id"], pid=gpu_data["pid"])
    
    # Initialize GPU from dataframes
    gpu.init_from_dfs(
        gpu_data["coll_info"],
        gpu_data["coll_kernels"],
        gpu_data["p2p_kernels"],
        gpu_data["comm_data"]
    )
    
    # Merge streams if requested
    if merged_streams:
        gpu.merge_streams()
    
    # Update init counter with lock to prevent race conditions
    with counter_lock:
        init_counter.value += 1
    
    # Generate and write goal file
    output_file = output_dir / f"rank_{rank}.goal"
    with open(output_file, "w") as f:
        f.write(f"rank {rank} {{\n")
        for line in gpu.generate_goal_lines(gpu_id2goal_rank, nic=nic):
            f.write(f"{line}\n")
        f.write("}\n")
    
    # Update generation counter with lock
    with counter_lock:
        gen_counter.value += 1
    
    return rank


def write_goal_for_gpu(args_tuple):
    """
    Worker function for parallel goal file writing.
    Writes goal for a single GPU to a separate file.
    
    Args:
        args_tuple: (gpu, rank, gpu_id2goal_rank, output_dir, merged_streams, nic)
    """
    gpu, rank, gpu_id2goal_rank, output_dir, merged_streams, nic = args_tuple
    
    if merged_streams:
        gpu.merge_streams()
    
    output_file = output_dir / f"rank_{rank}.goal"
    with open(output_file, "w") as f:
        f.write(f"rank {rank} {{\n")
        for line in gpu.generate_goal_lines(gpu_id2goal_rank, nic=nic):
            f.write(f"{line}\n")
        f.write("}\n")
    
    return rank


def concatenate_goal_files(output_dir: pathlib.Path, num_ranks: int, delete_parts: bool = True):
    """
    Concatenate individual rank goal files into a single output.goal file.
    
    Args:
        output_dir: Directory containing the rank_*.goal files
        num_ranks: Total number of ranks
        delete_parts: Whether to delete individual rank files after concatenation
    """
    output_file = output_dir / "output.goal"
    with open(output_file, "w") as out_f:
        out_f.write(f"num_ranks {num_ranks}\n")
        for rank in range(num_ranks):
            rank_file = output_dir / f"rank_{rank}.goal"
            with open(rank_file, "r") as rank_f:
                out_f.write(rank_f.read())
            if delete_parts:
                rank_file.unlink()
    return output_file


if __name__ == "__main__":
    # %%
    # get the path of the current script
    script_start_time = time.time()
    parser = argparse.ArgumentParser(description="Process some paths.")
    parser.add_argument("--trace_dir", type=str, required=True, help="Directory containing trace files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output files")
    parser.add_argument("--merged", action='store_true', help="Whether the streams are merged")
    parser.add_argument("--intermediate_results", action='store_true', help="Whether to save intermediate results")
    parser.add_argument("--dask", action='store_true', help="Whether to use dask for processing")
    parser.add_argument("--parallel_generation", action='store_true', help="Whether to generate goal files in parallel")
    parser.add_argument("--concatenate", action='store_true', help="Whether to concatenate all traces into one")
    parser.add_argument("--delete_parts", action='store_true', help="Whether to delete part files after concatenation")
    args = parser.parse_args()
    trace_dir = pathlib.Path(args.trace_dir).resolve()
    output_dir = pathlib.Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_streams = args.merged
    intermediate_results = args.intermediate_results
    parallel_generation = args.parallel_generation
    concatenate = args.concatenate
    delete_parts = args.delete_parts

    use_dask = args.dask
    if use_dask:
        from nsys_events_dask import *
        from tqdm.dask import TqdmCallback
        from functools import partial
        # Configure tqdm for batch job compatibility
        _tqdm_for_dask = partial(tqdm, file=sys.stderr, mininterval=0, dynamic_ncols=True)
    else:
        from nsys_events import *

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    traces = find_all_traces(trace_dir)
    nvtx_events = get_nvtx_events(traces)
    
    # nvtx_events is now Dict[str, pd.DataFrame] with pre-processed data
    # (parallel I/O, regex categorization, field extraction, and type conversion already done)

    comm_info, comm_ring_info, comm_tree_info, nvtx_events = get_communicator_info(nvtx_events)
    communicator_ids_numeric = [[i, comm_id] for i, comm_id in enumerate(comm_info["commId"].unique())]
    communicator_ids_numeric_df = pd.DataFrame(communicator_ids_numeric, columns=["comm_num_id", "commId"])

    # save comm_info, comm_ring_info, comm_tree_info to csv for debugging
    if intermediate_results:
        comm_info.to_csv(output_dir / "comm_info.csv", index=False)
        comm_ring_info.to_csv(output_dir / "comm_ring_info.csv", index=False)
        comm_tree_info.to_csv(output_dir / "comm_tree_info.csv", index=False)
    
    comm_data, coll_info, coll_kernels, p2p_kernels, nvtx_events = get_event_info(nvtx_events, comm_info)

    profiling_interval, nvtx_events = get_profiling_interval(nvtx_events)
    if intermediate_results:
        profiling_interval.to_csv(output_dir / "profiling_interval.csv", index=False)

    if intermediate_results:
        for data, name in zip([comm_data, coll_info, coll_kernels, p2p_kernels], ["comm_data", "coll_info", "coll_kernels", "p2p_kernels"]):
            curr_dir = output_dir / name
            curr_dir.mkdir(parents=True, exist_ok=True)
            for k, v in data.items():
                v.to_csv(curr_dir / f"{k[0]}_{k[1]}.csv", index=False)
    
    
    del nvtx_events
    
    # Pass traces directly - kernel events are read and processed in parallel tasks
    # Note: filter_time is called after association (not during)
    comm_data = associate_kernel_to_nvtx(comm_data, traces)

    comm_data = filter_time(profiling_interval, comm_data)
    comm_data = add_context_parallelism(comm_data)

    communicators, gpu_devices = construct_communicators(
        comm_info, comm_ring_info, comm_tree_info
    )

    if intermediate_results:
        curr_dir = output_dir / "comm_data_after"
        curr_dir.mkdir(parents=True, exist_ok=True)
        for k, v in comm_data.items():
            v.to_csv(curr_dir / f"{k[0]}_{k[1]}.csv", index=False)

    comm_ops = {
        "AllReduce": AllReduce,
        "AllGather": AllGather,
        "ReduceScatter": ReduceScatter,
        "Broadcast": Broadcast,
        "Reduce": Reduce,
        "Send": Send,
        "Recv": Recv,
    }
    
    # Build gpu_id2goal_rank mapping
    gpu_id2goal_rank = {gpu.gpu_id: i for i, gpu in enumerate(gpu_devices.values())}
    
    # Initialize npkit data
    init_data(
        "npkit_benchmark_results/clariden/npkit_data_summary_Simple.json",
        "npkit_benchmark_results/clariden/npkit_data_summary_LL.json",
    )

    def prepare_gpu_data(gpu_id, gpu):
        """Prepare dataframes for a single GPU, applying all transformations."""
        coll_info_gpu = coll_info[gpu_id].copy()
        coll_info_gpu["algo"] = coll_info_gpu["algo"].map(algo_mapping)
        coll_info_gpu["proto"] = coll_info_gpu["proto"].map(proto_mapping)
        
        coll_kernel_gpu = coll_kernels[gpu_id].copy()
        
        p2p_kernel_gpu = p2p_kernels[gpu_id].copy()
        p2p_kernel_gpu["proto"] = p2p_kernel_gpu["proto"].map(proto_mapping)
        # Vectorized bitwise operation
        hi32 = p2p_kernel_gpu["countHi32"].to_numpy(dtype=np.int64)
        lo32 = p2p_kernel_gpu["countLo32"].to_numpy(dtype=np.int64)
        p2p_kernel_gpu["count"] = (hi32 << 32) | lo32
        p2p_kernel_gpu.drop(columns=["countHi32", "countLo32"], inplace=True)
        
        comm_data_gpu = comm_data[gpu_id].copy()
        comm_data_gpu = comm_data_gpu.merge(communicator_ids_numeric_df, on="commId", how="left")
        comm_data_gpu["collective"] = comm_data_gpu["collective"].map(comm_ops)
        comm_data_gpu["communicator"] = comm_data_gpu["commId"].map(communicators)
        # Vectorized context_label calculation
        parallelism_vals = comm_data_gpu["parallelism"].map(context_labels).fillna(0).astype(np.int64)
        comm_num_ids = comm_data_gpu["comm_num_id"].to_numpy(dtype=np.int64)
        comm_data_gpu["context_label"] = parallelism_vals.to_numpy() + comm_num_ids * 100
        comm_data_gpu.drop(columns=["commId", "parallelism", "comm_num_id"], inplace=True)
        
        return {
            "coll_info": coll_info_gpu,
            "coll_kernels": coll_kernel_gpu,
            "p2p_kernels": p2p_kernel_gpu,
            "comm_data": comm_data_gpu,
            "node_id": gpu.node_id,
            "pid": gpu.pid,
        }
    
    if parallel_generation:
        # Parallel generation: prepare data, then use multiprocessing for init + goal generation
        logger.info("preparing GPU data for parallel generation")
        
        # Create shared counters for progress tracking
        manager = multiprocessing.Manager()
        init_counter = manager.Value('i', 0)
        gen_counter = manager.Value('i', 0)
        counter_lock = manager.Lock()
        
        # Prepare all GPU data in main process - stored in dict for initializer
        gpu_data_dict = {}
        task_list = []
        for gpu_id, gpu in tqdm(gpu_devices.items(), desc="Preparing GPU data"):
            gpu_data_dict[gpu_id] = prepare_gpu_data(gpu_id, gpu)
            rank = gpu_id2goal_rank[gpu.gpu_id]
            # Only pass small args per task - large data accessed via globals
            task_list.append((gpu_id, rank, 0, init_counter, gen_counter, counter_lock))
        
        time_finish_prep = time.time()
        logger.info(f"Data preparation time: {time_finish_prep - script_start_time:.2f} seconds")
        
        # Now run init + goal generation in parallel processes
        num_workers = min(os.cpu_count() or 1, len(gpu_devices))
        logger.info(f"initializing GPUs and generating goal files in parallel with {num_workers} workers")
        total_gpus = len(task_list)
        
        # Use initializer to set up shared data once per worker (copy-on-write friendly)
        with multiprocessing.Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(gpu_data_dict, gpu_id2goal_rank, output_dir, merged_streams)
        ) as pool:
            # Start async map - only small args are pickled now!
            async_result = pool.map_async(init_and_generate_goal_for_gpu, task_list)
            
            # Create dual progress bars
            # Init bar uses lighter/dimmer style, Gen bar uses solid style
            init_bar = tqdm(total=total_gpus, desc="Initializing ", position=0, 
                           bar_format='{desc}: {bar}| {n_fmt}/{total_fmt}',
                           colour='cyan', leave=True)
            gen_bar = tqdm(total=total_gpus, desc="Generating   ", position=1,
                          bar_format='{desc}: {bar}| {n_fmt}/{total_fmt}',
                          colour='green', leave=True)
            
            # Monitor progress until all tasks complete
            while not async_result.ready():
                # Update bars based on shared counters
                current_init = init_counter.value
                current_gen = gen_counter.value
                
                init_bar.n = current_init
                gen_bar.n = current_gen
                init_bar.refresh()
                gen_bar.refresh()
                
                time.sleep(0.1)
            
            # Final update to capture any last increments missed by the loop
            init_bar.n = init_counter.value
            gen_bar.n = gen_counter.value
            init_bar.refresh()
            gen_bar.refresh()
            init_bar.close()
            gen_bar.close()
            
            # Get results (will raise if any worker failed)
            results = async_result.get()
        
        time_finish_init = time.time()
        
        if concatenate:
            logger.info("concatenating goal files")
            goal_path = concatenate_goal_files(output_dir, len(gpu_devices), delete_parts=delete_parts)
        else:
            goal_path = output_dir
            logger.info(f"Goal files written to: {output_dir}/rank_*.goal")
    else:
        # Sequential mode: init GPUs then generate goals
        logger.info("initilizing GPU devices from dataframes")
        
        def init_one_gpu(gpu_id, gpu):
            gpu_data = prepare_gpu_data(gpu_id, gpu)
            gpu.init_from_dfs(
                gpu_data["coll_info"],
                gpu_data["coll_kernels"],
                gpu_data["p2p_kernels"],
                gpu_data["comm_data"]
            )
            return gpu_id
        
        if use_dask:
            from dask import delayed
            import dask
            tasks = [delayed(init_one_gpu)(gpu_id, gpu) for gpu_id, gpu in gpu_devices.items()]
            with TqdmCallback(desc="init GPUs", tqdm_class=_tqdm_for_dask):
                dask.compute(*tasks, scheduler="threads")
        else:
            for gpu_id, gpu in tqdm(gpu_devices.items()):
                init_one_gpu(gpu_id, gpu)

        time_finish_init = time.time()

        # Sequential write: single output file
        goal_path = output_dir / "output.goal"
        with open(goal_path, "w") as f:
            logger.info("writing goal file")
            gpus = gpu_devices.values()
            f.write(f"num_ranks {len(gpus)}\n")
            for gpu in tqdm(gpus):
                if merged_streams:
                    gpu.merge_streams()
                f.write(f"rank {gpu_id2goal_rank[gpu.gpu_id]} {{\n")
                for line in gpu.generate_goal_lines(gpu_id2goal_rank, nic=0):
                    f.write(f"{line}\n")
                f.write("}\n")
    
    script_finish_time = time.time()
    logger.info(f"Time to initialize GPU devices: {time_finish_init - script_start_time:.2f} seconds")
    logger.info(f"Time to generate goal file: {script_finish_time - time_finish_init:.2f} seconds")
    logger.info(f"Total script time: {time.time() - script_start_time:.2f} seconds")
    if isinstance(goal_path, pathlib.Path) and goal_path.is_file():
        logger.info(f"Goal file written to: {goal_path}")
# %%
