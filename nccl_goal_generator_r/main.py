#%%
from nccl_comm import *
from nccl_primitives import *
from nsys_events import *
from tqdm import tqdm
import pandas as pd
import logging
from typing import Dict, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor
import aiofiles
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


#%%
def construct_communicators(comm_info: pd.DataFrame, comm_ring_info: pd.DataFrame, comm_tree_info: pd.DataFrame) -> Tuple[Dict[str, Communicator], Dict[Tuple[str, int], GPUDevice]]:
    logger.info("constructing communicator objects")

    # construct GPUDevice objects
    gpus_df = comm_info[["nodeId", "pid"]].drop_duplicates()
    gpu_devices = {(row['nodeId'], row['pid']): GPUDevice(i) for i, (_, row) in enumerate(gpus_df.iterrows())}
    comm_info = comm_info.copy()
    comm_info["gpu"] = comm_info.apply(lambda row: gpu_devices[(row['nodeId'], row['pid'])], axis=1)
    comm_gpus_df = comm_info.sort_values("rank").groupby(["commId"]).aggregate({"gpu": list}).reset_index()
    communicators = {row["commId"]: Communicator(row["commId"], row["gpu"]) for _, row in comm_gpus_df.iterrows()}
    
    comm_ring_info = comm_ring_info.sort_values("channelId")
    for _, row in comm_ring_info.iterrows():
        communicators[row['commId']].add_ring_topo(
            row['myRank'], row['prevRank'], row['nextRank']
        )
    
    comm_tree_info = comm_tree_info.sort_values("channelId")
    for _, row in comm_tree_info.iterrows():
        children = [c for c in [row['child1Rank'], row['child2Rank'], row['child3Rank']] if c >= 0]
        communicators[row['commId']].add_tree_topo(
            row['myRank'], row['parentRank'], children
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

context_labels = {
    "Other" : 0,
    "PP": 1,
    "DP": 2
}

def construct_collectives(
    gpu_devices: Dict[Tuple[str, int], GPUDevice],
    communicators: Dict[str, Communicator],
    coll_info: pd.DataFrame,
    coll_kernels: pd.DataFrame,
    comm_data: pd.DataFrame,
    comm_info: pd.DataFrame) -> None:
    coll_info = coll_info.copy()
    
    coll_info["collInfo"] = coll_info.apply(
        lambda row: CollInfo(
            root_rank=row['root'], red_op=row['redOp'], algo=algo_mapping[row['algo']], proto=proto_mapping[row['proto']],
            data_size=row['data_size'], type_size=row['type_size'], chunk_size=row['chunkSize'], chunk_count=row['chunkCount'],
            chunk_steps=row['chunkSteps'], slice_steps=row['sliceSteps'], step_size=row['stepSize']
        ), axis=1
    )
    
    coll_kernels = coll_kernels.copy()
    coll_kernels["chnlInfo"] = coll_kernels.sort_values(["association", "workOffset"]).apply(
        lambda row: CollChnlInfo(
            n_warps=row['nWarps'], count=row['count'], chunk_count=row['chunkCount'],
            work_count=row['workCount'], last_chunk_count=row['lastChunkCount'], work_offset=row['workOffset'],
            send_buff=row['sendbuff'], recv_buff=row['recvbuff']
        ), axis=1
    )
    coll_kernels = coll_kernels.groupby("association").agg({"chnlInfo": list}).reset_index()
    
    coll_info = coll_info[["association", "collInfo"]].merge(
        coll_kernels, on="association", how="left"
    ).merge(
        comm_data, left_on="association", right_on="eventId", how="left"
    ).merge(
        comm_info[["nodeId", "commHash", "commId"]].drop_duplicates(),
        on=["nodeId", "commHash"], how="left"
    ).drop(columns=["association", "text"])
    
    coll_info["gpu"] = coll_info.apply(lambda row: gpu_devices[(row['nodeId'], row['pid'])], axis=1)
    coll_info["comm"] = coll_info.apply(lambda row: communicators[row['commId']], axis=1)
    collective_ops = {
        "AllReduce": AllReduce,
        "AllGather": AllGather,
        "ReduceScatter": ReduceScatter,
        "Broadcast": Broadcast,
        "Reduce": Reduce
    }
    coll_info["context_label"] = coll_info.apply(lambda row: context_labels.get(row['parallelism'], 0), axis=1)
    coll_info["collOp"] = coll_info.apply(lambda row: collective_ops[row['collective']](row['gpu'], row['comm'], row['collInfo'], row['chnlInfo'], row['context_label']), axis=1)
    for _, row in coll_info.iterrows():
        row['gpu'].add_collective(row['stream'], row['collOp'], row['start'], row['end'])

def construct_p2p(
    gpu_devices: Dict[Tuple[str, int], GPUDevice],
    communicators: Dict[str, Communicator],
    p2p_kernels: pd.DataFrame,
    comm_data: pd.DataFrame,
    comm_info: pd.DataFrame) -> None:
    p2p_kernels = p2p_kernels[["Bytes", "nWarps", "peer", "proto", "countHi32", "countLo32", "chunkSize", "association"]].merge(
        comm_data, left_on="association", right_on="eventId", how="left"
    ).merge(
        comm_info[["nodeId", "commHash", "commId"]].drop_duplicates(),
        on=["nodeId", "commHash"], how="left"
    )
    p2p_ops = {
        "Send": Send,
        "Recv": Recv
    }
    p2p_kernels['gpu'] = p2p_kernels.apply(lambda row: gpu_devices[(row['nodeId'], row['pid'])], axis=1)
    p2p_kernels['comm'] = p2p_kernels.apply(lambda row: communicators[row['commId']], axis=1)
    p2p_kernels['context_label'] = p2p_kernels.apply(lambda row: context_labels.get(row['parallelism'], 0), axis=1)
    p2p_kernels["p2pOp"] = p2p_kernels.apply(lambda row: p2p_ops[row['collective']](row["gpu"], row["Bytes"], row["peer"], row["comm"], row["chunkSize"], row['context_label']), axis=1)
    
    for _, row in p2p_kernels.iterrows():
        row['gpu'].add_collective(row["stream"], row['p2pOp'], row['start'], row['end'])


if __name__ == "__main__":
#%%
    traces = find_all_traces("../traces/Llama70B_N64_GPU256_TP1_PP8_DP32_70B_BS32/sqlite")
    kernel_events = get_kernel_events(traces)
    nvtx_events = get_nvtx_events(traces)
    comm_info, comm_ring_info, comm_tree_info = get_communicator_info(nvtx_events)
    profiling_interval = get_profiling_interval(nvtx_events)
    comm_data, coll_info, coll_kernels, p2p_kernels = get_event_info(nvtx_events, profiling_interval)
    comm_data, kernel_events = associate_kernel_to_nvtx(comm_data, kernel_events, profiling_interval)
    comm_data = add_context_parallelism(comm_data)
    
    communicators, gpu_devices = construct_communicators(comm_info, comm_ring_info, comm_tree_info)
    construct_collectives(gpu_devices, communicators, coll_info, coll_kernels, comm_data, comm_info)
    construct_p2p(gpu_devices, communicators, p2p_kernels, comm_data, comm_info)

    gpu2goal_rank = {gpu: i for i, gpu in enumerate(gpu_devices.values())}
    gpu2node = {gpu: gpu_id[0] for gpu_id, gpu in gpu_devices.items()}

    init_data("npkit_benchmark_results/ault/npkit_data_summary_Simple.json", "npkit_benchmark_results/ault/npkit_data_summary_LL.json")
    
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
    with open("trace.goal", "w") as f:
        logger.info("writing goal file")
        for gpu in tqdm(gpu_devices.values()):
            goal_gpu, _ = gpu.generate_goal(gpu2goal_rank, gpu2node, nic=0)
            lines = goal_gpu.generate_lines()
            f.write(f"rank {gpu2goal_rank[gpu]} {{\n")
            for line in lines:
                f.write(f" {line}\n")
            f.write("}\n")
            # result = f"rank {gpu2goal_rank[gpu]} {{\n {goal_gpu}\n}}\n"
            # f.write(result)
# %%
