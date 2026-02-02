# %%
from nccl_comm import *
from nccl_primitives import *
from nsys_events import *
from tqdm import tqdm
import pandas as pd
import logging
from typing import Dict, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor
# import aiofiles
from collections import defaultdict
import json
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# %%
def construct_communicators(comm_info: dict):
    communicators = {}
    gpu_devices = {}
    for comm_id, comm_data in comm_info.items():
        curr_comm_gpus = []
        for rank, rank_data in comm_data["rank_To_rankInfo"].items():
            gpu_id = rank_data["gpuId"]
            node_id = rank_data["goal_rank"]
            gpu_devices.setdefault(gpu_id, GPUDevice(gpu_id, node_id))
            curr_comm_gpus.append((int(rank), gpu_devices[gpu_id]))
        curr_comm_gpus.sort(key=lambda x: x[0])
        communicators[comm_id] = Communicator(comm_id, [gpu for _, gpu in curr_comm_gpus])
        for rank, rank_data in comm_data["rank_To_rankInfo"].items():
            gpu_id = rank_data["gpuId"]
            comm = communicators[comm_id]
            chnl_info = rank_data["channel_info"]
            for ring_info in chnl_info["Ring"]:
                comm.add_ring_topo(
                    int(rank), int(ring_info["previous_rank"]), int(ring_info["next_rank"])
                )
            for tree_info in chnl_info["Tree"]:
                children = [int(c) for c in (tree_info[f"child_{i}_rank"] for i in range(1, 4))]
                comm.add_tree_topo(
                    int(rank), int(tree_info["parent_rank"]), [c for c in children if c >= 0]
                )
    return communicators, gpu_devices

# %%
with open("nccl_example/example_allgather/results/nsys_events_intermediate_output.json", "r") as f:
    text = f.read()
    fields = text.split("\n\n")
    data = [json.loads(field) for field in fields if field.strip()]
    HostName_To_GoalRank, Comm_Info, CUPTI_Kernel_Results, NCCL_Events, Comm_Init_Events = data

# %%
communicators, gpu_devices = construct_communicators(Comm_Info)

# %%
collective_ops = { # make all collectives as AllReduce for testing
    "AllReduce": AllReduce,
    "AllGather": AllGather,
    "ReduceScatter": ReduceScatter,
    "Broadcast": Broadcast,
    "Reduce": Reduce
}
algo_mapping = {
    0: CollAlgo.TREE,
    1: CollAlgo.RING,
}
proto_mapping = {
    0: NCCLProto.LL,
    1: NCCLProto.LL128,
    2: NCCLProto.SIMPLE,
}

def construct_collectives(
        gpu_devices: Dict[int, GPUDevice],
        communicators: Dict[str, Communicator],
        collectives: dict
):
    flattened_collectives = {}
    for node_id, node_collectives in collectives.items():
        for gpu_id, gpu_collectives in node_collectives.items():
            flattened_collectives[int(gpu_id)] = gpu_collectives
    
    for gpu_id, gpu_collectives in flattened_collectives.items():
        for stream_id, stream_collectives in gpu_collectives.items():
            for coll in stream_collectives:
                chnl_infos = []
                for chnl_info in coll["elems"]:
                    chnl_infos.append(
                        CollChnlInfo(
                            3,
                            chnl_info["count"],
                            chnl_info["chunkCount"],
                            chnl_info["workCount"],
                            chnl_info["lastChunkCount"],
                            chnl_info["workOffset"],
                            chnl_info["sendbuff"],
                            chnl_info["recvbuff"]
                        )
                    )
                coll_info = CollInfo(
                    coll.get("root_rank", 0),
                    coll.get("redOp", 0),
                    algo_mapping[int(coll["algorithm"])],
                    proto_mapping[int(coll["protocol"])],
                    coll["data_size"],
                    coll["type_size"],
                    # -1,
                    # -1,
                    coll["chunkSteps"],
                    coll["sliceSteps"],
                    coll["stepSize"]
                )
                collective = collective_ops[coll["event_type"]](
                    gpu_devices[gpu_id],
                    communicators[coll["commId"]],
                    coll_info,
                    chnl_infos,
                    0
                )
                gpu_devices[gpu_id].add_collective(stream_id, collective, coll["ts_gpu_start"], coll["ts_gpu_end"])


# %%
with open("nccl_example/example_allgather/results/nsys_events_merged_output.json", "r") as f:
    coll_data = json.load(f)
construct_collectives(gpu_devices, communicators, coll_data)

# %%
init_data("npkit_benchmark_results/ault/npkit_data_summary_Simple.json", "npkit_benchmark_results/ault/npkit_data_summary_LL.json")

# %%
gpu2goal_rank = {gpu: i for i, gpu in enumerate(g for g in gpu_devices.values() if len(g.streams) > 0)}

with open("trace_allgather.goal", "w") as f:
    logger.info("writing goal file")
    gpus = [gpu for gpu in gpu_devices.values() if len(gpu.streams) > 0]
    f.write(f"num_ranks {len(gpus)}\n")
    for gpu in tqdm(gpus):
        f.write(f"rank {gpu2goal_rank[gpu]} {{\n")
        for line in gpu.generate_goal_lines(gpu2goal_rank, nic=0):
            f.write(f"{line}\n")
        f.write("}\n")


