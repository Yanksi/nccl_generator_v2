from nccl_comm import Communicator
from nccl_primitives import GPUDevice
import pandas as pd
import logging
from typing import Dict, Tuple
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def construct_communicators(comm_info: pd.DataFrame, comm_ring_info: pd.DataFrame, comm_tree_info: pd.DataFrame) -> Dict[Tuple[str, int, int], Communicator]:
    logger.info("constructing communicator objects")

    # construct GPUDevice objects
    gpus_df = comm_info[["nodeId", "pid"]].drop_duplicates()
    gpu_devices = {(row['nodeId'], row['pid']): GPUDevice(i) for i, (_, row) in enumerate(gpus_df.iterrows())}
    comm_info = comm_info.copy()
    comm_info["gpu"] = comm_info.apply(lambda row: gpu_devices[(row['nodeId'], row['pid'])], axis=1)
    comm_gpus_df = comm_info.sort_values("rank").groupby(["commId"]).aggregate({"gpu": list})
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
    
    return communicators