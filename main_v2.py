# %%
import logging

# import aiofiles
import pathlib
from typing import Dict, Tuple

import pandas as pd
from tqdm import tqdm

from nccl_comm import *
from nccl_primitives import *
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
        (row["nodeId"], row["pid"]): GPUDevice(i, int(row["nodeId"]))
        for i, (_, row) in enumerate(gpus_df.iterrows())
    }
    comm_info = comm_info.copy()
    comm_info["gpu"] = comm_info.apply(
        lambda row: gpu_devices[(row["nodeId"], row["pid"])], axis=1
    )
    comm_gpus_df = (
        comm_info.sort_values("rank")
        .groupby(["commId"])
        .aggregate({"gpu": list})
        .reset_index()
    )
    communicators = {
        row["commId"]: Communicator(row["commId"], row["gpu"])
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
    args = parser.parse_args()
    trace_dir = pathlib.Path(args.trace_dir).resolve()
    output_dir = pathlib.Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_streams = args.merged
    intermediate_results = args.intermediate_results
    use_dask = args.dask
    if use_dask:
        from nsys_events_dask import *
        from dask.diagnostics import ProgressBar
    else:
        from nsys_events import *

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
    
    kernel_events = get_kernel_events(traces)
    comm_data = associate_kernel_to_nvtx(comm_data, kernel_events)
    
    del kernel_events

    communicators, gpu_devices = construct_communicators(
        comm_info, comm_ring_info, comm_tree_info
    )

    if intermediate_results:
        curr_dir = output_dir / "comm_data_after"
        curr_dir.mkdir(parents=True, exist_ok=True)
        for k, v in comm_data.items():
            v.to_csv(curr_dir / f"{k[0]}_{k[1]}.csv", index=False)

    comm_data = filter_time(profiling_interval, comm_data)
    comm_data = add_context_parallelism(comm_data)


    logger.info("initilizing GPU devices from dataframes")
    comm_ops = {
        "AllReduce": AllReduce,
        "AllGather": AllGather,
        "ReduceScatter": ReduceScatter,
        "Broadcast": Broadcast,
        "Reduce": Reduce,
        "Send": Send,
        "Recv": Recv,
    }
    
    def init_one_gpu(gpu_id, gpu):
        coll_info_gpu = coll_info[gpu_id]
        coll_info_gpu["algo"] = coll_info_gpu["algo"].map(algo_mapping)
        coll_info_gpu["proto"] = coll_info_gpu["proto"].map(proto_mapping)
        
        coll_kernel_gpu = coll_kernels[gpu_id]
        
        p2p_kernel_gpu = p2p_kernels[gpu_id]
        p2p_kernel_gpu["proto"] = p2p_kernel_gpu["proto"].map(proto_mapping)
        p2p_kernel_gpu["count"] = p2p_kernel_gpu[["countHi32", "countLo32"]].apply(
            lambda row: row["countHi32"] << 32 | row["countLo32"], axis=1
        )
        p2p_kernel_gpu.drop(columns=["countHi32", "countLo32"], inplace=True)
        
        comm_data_gpu = comm_data[gpu_id]
        comm_data_gpu = comm_data_gpu.merge(communicator_ids_numeric_df, on="commId", how="left")
        comm_data_gpu["collective"] = comm_data_gpu["collective"].map(comm_ops)
        comm_data_gpu["communicator"] = comm_data_gpu["commId"].map(communicators)
        comm_data_gpu["context_label"] = comm_data_gpu.apply(
            lambda row: context_labels.get(row["parallelism"], 0) + row["comm_num_id"] * 100, axis=1
        )
        comm_data_gpu.drop(columns=["commId", "parallelism", "comm_num_id"], inplace=True)
        
        gpu.init_from_dfs(coll_info_gpu, coll_kernel_gpu, p2p_kernel_gpu, comm_data_gpu)
        return gpu_id
    
    if use_dask:
        from dask import delayed
        import dask
        tasks = [delayed(init_one_gpu)(gpu_id, gpu) for gpu_id, gpu in gpu_devices.items()]
        with ProgressBar():
            dask.compute(*tasks, scheduler="threads")
    else:
        for gpu_id, gpu in tqdm(gpu_devices.items()):
            init_one_gpu(gpu_id, gpu)

    gpu2goal_rank = {gpu: i for i, gpu in enumerate(gpu_devices.values())}
    gpu2node = {gpu: gpu_id[0] for gpu_id, gpu in gpu_devices.items()}

    init_data(
        "npkit_benchmark_results/clariden/npkit_data_summary_Simple.json",
        "npkit_benchmark_results/clariden/npkit_data_summary_LL.json",
    )

    # async def write_goals_buffered():
    #     logger.info("writing goal file")
    #     write_tasks = []
    #     async with aiofiles.open("trace.goal", "w") as f:
    #         for gpu in tqdm(gpu_devices.values()):
    #             goal_gpu, _ = gpu.generate_goal(gpu2goal_rank, gpu2node, nic=0)
    #             result = f"rank {gpu2goal_rank[gpu]} {{\n {goal_gpu}\n}}\n"
    #             write_tasks.append(f.write(result))
    #         await asyncio.gather(*write_tasks)
    # asyncio.run(write_goals_buffered())
    time_finish_init = time.time()
    goal_path = output_dir / "output.goal"
    with open(goal_path, "w") as f:
        logger.info("writing goal file")
        gpus = gpu_devices.values()
        f.write(f"num_ranks {len(gpus)}\n")
        for gpu in tqdm(gpus):
            if merged_streams:
                gpu.merge_streams()
            # gpu.merge_streams()
            f.write(f"rank {gpu2goal_rank[gpu]} {{\n")
            for line in gpu.generate_goal_lines(gpu2goal_rank, nic=0):
                f.write(f"{line}\n")
            f.write("}\n")
    script_finish_time = time.time()
    logger.info(f"Time to initialize GPU devices: {time_finish_init - script_start_time:.2f} seconds")
    logger.info(f"Time to generate goal file: {script_finish_time - time_finish_init:.2f} seconds")
    logger.info(f"Total script time: {time.time() - script_start_time:.2f} seconds")
    logger.info(f"Goal file written to: {goal_path}")
# %%
