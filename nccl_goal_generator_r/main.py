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

    communicators = {}
    for _, row in comm_info.iterrows():
        key = (row['nodeId'], row['commId'], row['pid'])
        communicators[key] = Communicator(
            comm_hash=row['commHash'],
            comm_id=row['commId'],
            rank=row['rank'],
            n_ranks=row['nRanks'],
            pid=row['pid'],
            node_id=row['nodeId']
        )
    
    for _, row in comm_ring_info.iterrows():
        key = (row['nodeId'], row['commId'], row['pid'])
        if key in communicators:
            communicators[key].add_ring(
                channel_id=row['channelId'],
                prev_rank=row['prevRank'],
                my_rank=row['myRank'],
                next_rank=row['nextRank']
            )
    
    for _, row in comm_tree_info.iterrows():
        key = (row['nodeId'], row['commId'], row['pid'])
        if key in communicators:
            communicators[key].add_tree(
                channel_id=row['channelId'],
                child_ranks=[row['child_1_rank'], row['child_2_rank'], row['child_3_rank']],
                my_rank=row['my_rank'],
                parent_rank=row['parent_rank']
            )
    
    return communicators