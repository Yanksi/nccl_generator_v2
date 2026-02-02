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


def parse_buffer_size(size_str: str) -> int:
    """
    Parse a buffer size string into bytes.
    Supports formats like: '1MB', '512KB', '8192', '1M', '512K'
    """
    if size_str is None:
        return None
    
    size_str = size_str.strip().upper()
    
    # Check for unit suffixes
    if size_str.endswith('MB'):
        return int(float(size_str[:-2]) * 1024 * 1024)
    elif size_str.endswith('KB'):
        return int(float(size_str[:-2]) * 1024)
    elif size_str.endswith('GB'):
        return int(float(size_str[:-2]) * 1024 * 1024 * 1024)
    elif size_str.endswith('M'):
        return int(float(size_str[:-1]) * 1024 * 1024)
    elif size_str.endswith('K'):
        return int(float(size_str[:-1]) * 1024)
    elif size_str.endswith('B'):
        return int(size_str[:-1])
    else:
        # Assume plain number in bytes
        return int(size_str)


# Global variables for worker processes (set via Pool initializer for copy-on-write efficiency)
_WORKER_GPU_DATA = None
_WORKER_GPU_ID2GOAL_RANK = None
_WORKER_OUTPUT_DIR = None
_WORKER_MERGED_STREAMS = None
_WORKER_WRITE_BUFFER_SIZE = None
_WORKER_MERGE_SUCCESS = None  # Shared array for tracking merge success per GPU
_WORKER_MERGE_BARRIER = None  # Barrier for synchronizing merge phase


def _init_worker(gpu_data_dict, gpu_id2goal_rank, output_dir, merged_streams, write_buffer_size,
                 merge_success_array=None, merge_barrier=None):
    """
    Initialize worker process with shared data.
    This is called once per worker at fork time, enabling copy-on-write memory sharing.
    """
    global _WORKER_GPU_DATA, _WORKER_GPU_ID2GOAL_RANK, _WORKER_OUTPUT_DIR, _WORKER_MERGED_STREAMS
    global _WORKER_WRITE_BUFFER_SIZE, _WORKER_MERGE_SUCCESS, _WORKER_MERGE_BARRIER
    _WORKER_GPU_DATA = gpu_data_dict
    _WORKER_GPU_ID2GOAL_RANK = gpu_id2goal_rank
    _WORKER_OUTPUT_DIR = output_dir
    _WORKER_MERGED_STREAMS = merged_streams
    _WORKER_WRITE_BUFFER_SIZE = write_buffer_size
    _WORKER_MERGE_SUCCESS = merge_success_array
    _WORKER_MERGE_BARRIER = merge_barrier


def process_gpu_chunk(args_tuple):
    """
    Worker function for parallel GPU initialization and goal generation.
    Processes a chunk of GPUs (not just one) for better load balancing.
    Uses global data set via initializer (copy-on-write friendly).
    
    Two-phase stream merging:
    - Phase 1: Try merge_streams() for all GPUs in chunk, record success/failure
    - Barrier: Wait for all workers to complete phase 1
    - Phase 2: If ALL GPUs across all workers succeeded, apply merged streams; otherwise skip
    
    Args:
        args_tuple: (gpu_chunk, nic, init_counter, gen_counter, counter_lock)
        gpu_chunk: List of (gpu_id, rank) tuples for this worker to process
    """
    gpu_chunk, nic, init_counter, gen_counter, counter_lock = args_tuple
    
    # Access data from globals (copy-on-write, no pickle overhead)
    gpu_data_dict = _WORKER_GPU_DATA
    gpu_id2goal_rank = _WORKER_GPU_ID2GOAL_RANK
    output_dir = _WORKER_OUTPUT_DIR
    merged_streams = _WORKER_MERGED_STREAMS
    merge_success_array = _WORKER_MERGE_SUCCESS
    merge_barrier = _WORKER_MERGE_BARRIER
    
    # Phase 1: Initialize all GPUs in chunk and attempt stream merging
    gpus = []  # List of (gpu, rank, merged_stream_or_none)
    for gpu_id, rank in gpu_chunk:
        gpu_data = gpu_data_dict[gpu_id]
        
        # Reconstruct GPUDevice in this process
        gpu = GPUDevice(rank=rank, node_id=gpu_data["node_id"], pid=gpu_data["pid"])
        
        # Initialize GPU from dataframes
        gpu.init_from_dfs(
            gpu_data["coll_info"],
            gpu_data["coll_kernels"],
            gpu_data["p2p_kernels"],
            gpu_data["comm_data"]
        )
        
        # Try to merge streams if requested
        merged_stream = None
        if merged_streams:
            merged_stream = gpu.merge_streams()
            # Record success/failure in shared array (1 = success, 0 = failure)
            merge_success_array[rank] = 1 if merged_stream is not None else 0
        
        gpus.append((gpu, rank, merged_stream))
        
        # Update init counter
        with counter_lock:
            init_counter.value += 1
    
    # Barrier: Wait for all workers to complete Phase 1 (merge attempts)
    if merged_streams and merge_barrier is not None:
        merge_barrier.wait()
    
    # Phase 2: Check if ALL GPUs succeeded in merging
    all_merge_success = True
    if merged_streams and merge_success_array is not None:
        # Check all entries in the shared array
        all_merge_success = all(merge_success_array[i] == 1 for i in range(len(merge_success_array)))
    
    # Phase 3: Generate goal files
    results = []
    for gpu, rank, merged_stream in gpus:
        # Apply merged streams only if ALL GPUs succeeded
        if merged_streams and all_merge_success and merged_stream is not None:
            gpu.streams = {"merged_stream": merged_stream}
        # else: keep original streams
        
        # Generate and write goal file
        output_file = output_dir / f"rank_{rank}.goal"
        buffering = _WORKER_WRITE_BUFFER_SIZE if _WORKER_WRITE_BUFFER_SIZE and _WORKER_WRITE_BUFFER_SIZE > 0 else -1
        with open(output_file, "w", buffering=buffering) as f:
            f.write(f"rank {rank} {{\n")
            for line in gpu.generate_goal_lines(gpu_id2goal_rank, nic=nic):
                f.write(f"{line}\n")
            f.write("}\n")
        
        # Update generation counter
        with counter_lock:
            gen_counter.value += 1
        
        results.append(rank)
    
    return results


def write_goal_for_gpu(args_tuple):
    """
    Worker function for parallel goal file writing.
    Writes goal for a single GPU to a separate file.
    
    Args:
        args_tuple: (gpu, rank, gpu_id2goal_rank, output_dir, merged_streams, nic, write_buffer_size)
    """
    gpu, rank, gpu_id2goal_rank, output_dir, merged_streams, nic, write_buffer_size = args_tuple
    
    if merged_streams:
        gpu.streams = {"stream_merged": gpu.merge_streams()}

    output_file = output_dir / f"rank_{rank}.goal"
    buffering = write_buffer_size if write_buffer_size and write_buffer_size > 0 else -1
    with open(output_file, "w", buffering=buffering) as f:
        f.write(f"rank {rank} {{\n")
        for line in gpu.generate_goal_lines(gpu_id2goal_rank, nic=nic):
            f.write(f"{line}\n")
        f.write("}\n")
    
    return rank


def concatenate_goal_files(output_dir: pathlib.Path, num_ranks: int, delete_parts: bool = True, 
                           write_buffer_size: int = None):
    """
    Concatenate individual rank goal files into a single output.goal file.
    
    Args:
        output_dir: Directory containing the rank_*.goal files
        num_ranks: Total number of ranks
        delete_parts: Whether to delete individual rank files after concatenation
        write_buffer_size: Buffer size for file writes (None = system default)
    """
    output_file = output_dir / "output.goal"
    buffering = write_buffer_size if write_buffer_size and write_buffer_size > 0 else -1
    with open(output_file, "w", buffering=buffering) as out_f:
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
    parser.add_argument("--trace_dir", "-i", type=str, required=True, help="Directory containing trace files")
    parser.add_argument("--output_dir", "-o", type=str, required=True, help="Directory to save output files")
    parser.add_argument("--npkit_data_simple", "-s", type=str, required=True, help="Path to npkit data summary for Simple protocol")
    parser.add_argument("--npkit_data_ll", "-l", type=str, required=True, help="Path to npkit data summary for LL protocol")
    parser.add_argument("--merged", "-m", action='store_true', help="Whether the streams are merged")
    parser.add_argument("--parallel_generation", "-p", action='store_true', help="Whether to generate goal files in parallel")
    parser.add_argument("--concatenate", "-c", action='store_true', help="Whether to concatenate all traces into one")
    parser.add_argument("--delete_parts", "-r", action='store_true', help="Whether to delete part files after concatenation (will automatically enable concatenation)")
    parser.add_argument("--write_buffer_size", type=str, default=None, 
                        help="Write buffer size for goal file output (e.g., '1MB', '512KB', '8192'). "
                             "Larger buffers reduce I/O operations, useful for network storage.")
    parser.add_argument("--n_workers", "-w", type=int, default=os.cpu_count(), help="Maximum number of worker processes")

    args = parser.parse_args()
    trace_dir = pathlib.Path(args.trace_dir).resolve()
    output_dir = pathlib.Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_streams = args.merged
    parallel_generation = args.parallel_generation
    n_workers = args.n_workers
    concatenate = args.concatenate
    delete_parts = args.delete_parts
    if delete_parts:
        concatenate = True  # must concatenate if deleting parts
    write_buffer_size = parse_buffer_size(args.write_buffer_size)
    if write_buffer_size:
        logger.info(f"Using write buffer size: {write_buffer_size} bytes ({write_buffer_size / 1024:.1f} KB)")

    npkit_data_simple = args.npkit_data_simple
    npkit_data_ll = args.npkit_data_ll
    if parallel_generation:
        import dask
        dask.config.set(scheduler="processes", num_workers=n_workers)
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
    
    comm_data, coll_info, coll_kernels, p2p_kernels, nvtx_events = get_event_info(nvtx_events, comm_info)

    profiling_interval, nvtx_events = get_profiling_interval(nvtx_events)
    
    del nvtx_events
    
    kernel_events = get_kernel_events(traces)
    comm_data = associate_kernel_to_nvtx(comm_data, kernel_events)

    comm_data = filter_time(profiling_interval, comm_data)
    comm_data = add_context_parallelism(comm_data)

    communicators, gpu_devices = construct_communicators(
        comm_info, comm_ring_info, comm_tree_info
    )

    comm_ops = {
        "AllReduce": AllReduce,
        "AllGather": AllGather,
        "ReduceScatter": ReduceScatter,
        "Broadcast": Broadcast,
        "Reduce": Reduce,
        "Send": Send,
        "Recv": Recv,
    }

    comm_op_ids = {name: i for i, name in enumerate(comm_ops.keys())}
    comm_op_ids["Send"] = comm_op_ids["Recv"]
    
    # Build gpu_id2goal_rank mapping
    gpu_id2goal_rank = {gpu.gpu_id: i for i, gpu in enumerate(gpu_devices.values())}
    
    # Initialize npkit data
    init_data(
        npkit_data_simple,
        npkit_data_ll,
    )

    _MASK = (1 << sys.hash_info.width) - 1
    def hash_sequnces(*sequences):
        return [hash(elems) & _MASK for elems in zip(*sequences)]

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
        comm_data_gpu = comm_data_gpu.sort_values(["start"])
        
        # Vectorized context_label calculation
        parallelism_vals = comm_data_gpu["parallelism"].map(context_labels).fillna(0).astype(np.int64)
        # make sure one communicator is used by only one stream only
        for commId, group in comm_data_gpu.groupby("commId"):
            stream_ids = group["stream"].unique()
            if len(stream_ids) > 1:
                raise ValueError(f"Communicator {commId} used by multiple streams: {stream_ids}")
        
        # prepare identifier tag
        comm_num_ids = comm_data_gpu["comm_num_id"].to_numpy(dtype=np.int64)
        comm_data_gpu["comm_op_id"] = comm_data_gpu["collective"].map(comm_op_ids)
        comm_op_id = comm_data_gpu["comm_op_id"].to_numpy(dtype=np.int64)
        comm_seq_ids = comm_data_gpu.groupby(["commId", "comm_op_id"]).cumcount().to_numpy(dtype=np.int64)
        comm_seq_ids[comm_data_gpu["collective"].isin(["Send", "Recv"])] = 0  # reset seq_id for point-to-point ops
        comm_identifier = (np.array(hash_sequnces(comm_num_ids, comm_op_id, comm_seq_ids)) % 1000).astype(np.int64)

        comm_data_gpu["context_label"] = parallelism_vals.to_numpy() + comm_identifier * 100

        comm_data_gpu["collective"] = comm_data_gpu["collective"].map(comm_ops)
        comm_data_gpu["communicator"] = comm_data_gpu["commId"].map(communicators)

        comm_data_gpu.drop(columns=["commId", "parallelism", "comm_num_id", "comm_op_id"], inplace=True)
        
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
        # GPUs are chunked into n_workers chunks for coordinated stream merging
        logger.info("preparing GPU data for parallel generation")
        
        # Create shared counters for progress tracking
        manager = multiprocessing.Manager()
        init_counter = manager.Value('i', 0)
        gen_counter = manager.Value('i', 0)
        counter_lock = manager.Lock()
        
        # Create shared array for tracking merge success per GPU (used for coordinated merging)
        total_gpus = len(gpu_devices)
        merge_success_array = None
        merge_barrier = None
        if merged_streams:
            # Shared array: 1 = merge success, 0 = merge failure (indexed by rank)
            merge_success_array = multiprocessing.Array('i', [0] * total_gpus)
            # Barrier to synchronize all workers after merge phase
            merge_barrier = multiprocessing.Barrier(n_workers)
        
        # Prepare all GPU data in main process - stored in dict for initializer
        gpu_data_dict = {}
        gpu_list = []  # List of (gpu_id, rank) tuples
        for gpu_id, gpu in tqdm(gpu_devices.items(), desc="Preparing GPU data"):
            gpu_data_dict[gpu_id] = prepare_gpu_data(gpu_id, gpu)
            rank = gpu_id2goal_rank[gpu.gpu_id]
            gpu_list.append((gpu_id, rank))
        
        # Chunk GPUs into n_workers chunks (one chunk per worker)
        # This ensures coordinated stream merging across all GPUs
        def chunk_list(lst, n_chunks):
            """Split list into n roughly equal chunks."""
            chunk_size = len(lst) // n_chunks
            remainder = len(lst) % n_chunks
            chunks = []
            start = 0
            for i in range(n_chunks):
                # Distribute remainder across first 'remainder' chunks
                end = start + chunk_size + (1 if i < remainder else 0)
                chunks.append(lst[start:end])
                start = end
            chunks = [chunk for chunk in chunks if chunk]  # remove empty chunks
            return chunks
        
        gpu_chunks = chunk_list(gpu_list, n_workers)
        # Build task list: each task is a chunk of GPUs
        task_list = [
            (chunk, 0, init_counter, gen_counter, counter_lock)  # nic=0
            for chunk in gpu_chunks
        ]
        
        time_finish_prep = time.time()
        logger.info(f"Data preparation time: {time_finish_prep - script_start_time:.2f} seconds")
        
        # Now run init + goal generation in parallel processes
        logger.info(f"initializing GPUs and generating goal files in parallel with {min(n_workers, len(gpu_chunks))} workers")
        logger.info(f"Total GPUs: {total_gpus}, chunked into {len(gpu_chunks)} chunks")
        if merged_streams:
            logger.info("Stream merging enabled: coordinated across all GPUs (all-or-none)")
        
        # Use initializer to set up shared data once per worker (copy-on-write friendly)
        with multiprocessing.Pool(
            processes=min(n_workers, len(gpu_chunks)),
            initializer=_init_worker,
            initargs=(gpu_data_dict, gpu_id2goal_rank, output_dir, merged_streams, write_buffer_size,
                      merge_success_array, merge_barrier)
        ) as pool:
            # Start async map - each task processes a chunk of GPUs
            async_result = pool.map_async(process_gpu_chunk, task_list)
            
            # Create dual progress bars
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
        
        # Log merge result
        if merged_streams and merge_success_array is not None:
            all_merged = all(merge_success_array[i] == 1 for i in range(total_gpus))
            if all_merged:
                logger.info("Stream merging: SUCCESS - all GPUs merged their streams")
            else:
                failed_ranks = [i for i in range(total_gpus) if merge_success_array[i] == 0]
                logger.info(f"Stream merging: SKIPPED - {len(failed_ranks)} GPU(s) could not merge (ranks: {failed_ranks[:10]}{'...' if len(failed_ranks) > 10 else ''})")
        
        time_finish_init = time.time()
        
        if concatenate:
            logger.info("concatenating goal files")
            goal_path = concatenate_goal_files(output_dir, len(gpu_devices), delete_parts=delete_parts,
                                               write_buffer_size=write_buffer_size)
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
        
        if parallel_generation:
            from dask import delayed
            import dask
            tasks = [delayed(init_one_gpu)(gpu_id, gpu) for gpu_id, gpu in gpu_devices.items()]
            with TqdmCallback(desc="init GPUs", tqdm_class=_tqdm_for_dask):
                dask.compute(*tasks, scheduler="threads")
        else:
            for gpu_id, gpu in tqdm(gpu_devices.items()):
                init_one_gpu(gpu_id, gpu)

        time_finish_init = time.time()

        # Coordinated stream merging: all GPUs merge only if all can merge
        merged_streams_dict = {}
        if merged_streams:
            all_can_merge = True
            for gpu in gpu_devices.values():
                merged = gpu.merge_streams()
                if merged is None:
                    all_can_merge = False
                    logger.info(f"Stream merging: GPU {gpu.id} cannot merge streams")
                    break
                merged_streams_dict[gpu.gpu_id] = merged
            
            if all_can_merge:
                logger.info("Stream merging: SUCCESS - all GPUs merged their streams")
                for gpu in gpu_devices.values():
                    gpu.streams = {"stream_merged": merged_streams_dict[gpu.gpu_id]}
            else:
                logger.info("Stream merging: SKIPPED - not all GPUs could merge")

        # Sequential write: single output file
        goal_path = output_dir / "output.goal"
        buffering = write_buffer_size if write_buffer_size and write_buffer_size > 0 else -1
        with open(goal_path, "w", buffering=buffering) as f:
            logger.info("writing goal file")
            gpus = gpu_devices.values()
            f.write(f"num_ranks {len(gpus)}\n")
            for gpu in tqdm(gpus):
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
