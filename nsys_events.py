import logging
import os
import pathlib
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numba

# import modin.pandas as pd
import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger("nsys_events")
logging.basicConfig(level=logging.INFO)


def find_all_traces(directory):
    # find all trace files that ends with .sqlite
    return list(pathlib.Path(directory).rglob("*.sqlite"))


def extract_node_id(filename: str) -> str:
    """Extract node ID from trace filename. Supports both nid and profile_job_node_rank formats."""
    # Try nid format first (e.g., "profile_nid007654_123.sqlite")
    match = re.search(r"nid(\d+)", filename)
    if match:
        return match.group(1)
    # Try profile_job_node_rank format (e.g., "profile_6405_0_0.sqlite")
    match = re.search(r"profile_\d+_(\d+)_\d+\.sqlite", filename)
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract node ID from filename: {filename}")


def get_kernel_events(traces: List[os.PathLike]) -> pd.DataFrame:
    logger.info("querying for kernel events")
    kernel_dfs = []
    for trace_file in tqdm(traces):
        node_id = extract_node_id(trace_file.name)
        conn = sqlite3.connect(trace_file)
        df_tmp = pd.read_sql_query(
            "SELECT start, end, value, deviceId, streamId, globalPid / 0x1000000 % 0x1000000 AS pid FROM CUPTI_ACTIVITY_KIND_KERNEL cakk, StringIds si WHERE cakk.demangledName = si.id and si.value LIKE 'nccl%'",
            conn,
            dtype={
                "start": "Int64",
                "end": "Int64",
                "value": "string",
                "deviceId": "Int64",
                "streamId": "Int64",
                "pid": "Int64",
            },
        )
        conn.close()
        df_tmp["nodeId"] = node_id
        df_tmp["collective"] = df_tmp["value"].str.extract(r"ncclDevKernel_([a-zA-Z]+)")
        kernel_dfs.append(df_tmp)

    kernel_df = pd.concat(kernel_dfs, ignore_index=True, copy=False)
    kernel_df.drop(columns=["value"], inplace=True)
    # kernel_df['eventId'] = range(len(kernel_df))  # Add unique ID to each row
    # kernel_df["stream"] = kernel_df[["deviceId", "streamId"]].apply(lambda x: f"{x['deviceId']}-{x['streamId']}", axis=1)
    return kernel_df


def get_nvtx_events(traces: List[os.PathLike]) -> pd.DataFrame:
    logger.info("querying for nvtx events")
    nvtx_dfs = []
    for trace_file in tqdm(traces):
        node_id = extract_node_id(trace_file.name)
        conn = sqlite3.connect(trace_file)
        df_tmp = pd.read_sql_query(
            "SELECT start, end, text FROM NVTX_EVENTS",
            conn,
            dtype={"start": "Int64", "end": "Int64", "text": "string"},
        )
        conn.close()
        df_tmp["nodeId"] = node_id
        nvtx_dfs.append(df_tmp)

    nvtx_df = pd.concat(nvtx_dfs, ignore_index=True, copy=False)
    nvtx_df["eventId"] = range(len(nvtx_df))  # Add unique ID to each row
    return nvtx_df


def convert_numeric(data_frame: pd.DataFrame, signed_columns: List[str], unsigned_columns: List[str]) -> pd.DataFrame:
    # if safe:
    #     # get 19 least significant digits to fit in int64
    #     data_frame[columns] = (
    #         data_frame[columns].astype("string").str.slice(-19).astype("Int64")
    #     )
    # else:
    if len(signed_columns) > 0:
        data_frame[signed_columns] = data_frame[signed_columns].astype("Int64")
    if len(unsigned_columns) > 0:
        data_frame[unsigned_columns] = data_frame[unsigned_columns].astype("UInt64")
    return data_frame


def get_communicator_info(data: pd.DataFrame):
    logger.info("extracting communicator info")
    # extract available informations from the table
    comm_info_pattern = (
        r"commHash (0x[0-9a-f]+) commId (0x[0-9a-f]+) rank (\d+) nranks (\d+) pid (\d+)"
    )
    comm_info_pattern_matching_indexs = data["text"].str.match(comm_info_pattern)
    comm_info = data[comm_info_pattern_matching_indexs].copy()
    data = data[~comm_info_pattern_matching_indexs]
    
    if len(comm_info) > 0:
        # Standard NCCL 2.20 format with full init markers
        comm_info[["commHash", "commId", "rank", "nRanks", "pid"]] = comm_info[
            "text"
        ].str.extract(comm_info_pattern)
        comm_info = convert_numeric(comm_info, [], ["rank", "nRanks", "pid"])
        comm_info = comm_info.drop(
            columns=["text", "start", "end", "eventId"]
        ).drop_duplicates()
        comm_hash2id = comm_info[["nodeId", "commHash", "commId"]].drop_duplicates()
    else:
        # NCCL 2.28 format: no init markers, synthesize from collective calls
        logger.warning("No communicator init markers found - synthesizing from collective calls")
        
        # Extract commHash from collective calls
        coll_pattern = r"nccl[a-zA-Z]+\(\): commHash (0x[0-9a-f]+),.*pid (\d+)"
        coll_matches = data["text"].str.extract(coll_pattern)
        coll_matches.columns = ["commHash", "pid"]
        coll_matches = coll_matches.dropna()
        coll_matches["nodeId"] = data.loc[coll_matches.index, "nodeId"]
        coll_matches["pid"] = coll_matches["pid"].astype("Int64")
        
        # Get unique (nodeId, pid, commHash) combinations
        unique_gpus = coll_matches[["nodeId", "pid"]].drop_duplicates().reset_index(drop=True)
        unique_comms = coll_matches["commHash"].unique()
        
        # Synthesize comm_info
        comm_info_rows = []
        for commHash in unique_comms:
            # Use commHash as commId too (since we don't have separate commId)
            gpus_in_comm = coll_matches[coll_matches["commHash"] == commHash][["nodeId", "pid"]].drop_duplicates()
            nRanks = len(gpus_in_comm)
            for rank, (_, row) in enumerate(gpus_in_comm.sort_values(["nodeId", "pid"]).iterrows()):
                comm_info_rows.append({
                    "nodeId": row["nodeId"],
                    "commHash": commHash,
                    "commId": commHash,  # Use commHash as commId
                    "rank": rank,
                    "nRanks": nRanks,
                    "pid": row["pid"]
                })
        
        if comm_info_rows:
            comm_info = pd.DataFrame(comm_info_rows)
            comm_info = convert_numeric(comm_info, [], ["rank", "nRanks", "pid"])
        else:
            comm_info = pd.DataFrame(columns=["nodeId", "commHash", "commId", "rank", "nRanks", "pid"])
        
        comm_hash2id = comm_info[["nodeId", "commHash", "commId"]].drop_duplicates()

    logger.info("extracting communicator rings")
    comm_ring_pattern = (
        r"commHash (0x[0-9a-f]+) Rings \[(\d+)\] (\d+)->(\d+)->(\d+) pid (\d+)"
    )
    comm_ring_pattern_matching_indexs = data["text"].str.match(comm_ring_pattern)
    comm_ring_info = data[comm_ring_pattern_matching_indexs].copy()
    data = data[~comm_ring_pattern_matching_indexs]
    
    if len(comm_ring_info) > 0:
        comm_ring_info[
            ["commHash", "channelId", "prevRank", "myRank", "nextRank", "pid"]
        ] = comm_ring_info["text"].str.extract(comm_ring_pattern)
        comm_ring_info = convert_numeric(comm_ring_info, [], ["channelId", "prevRank", "myRank", "nextRank", "pid"])
        comm_ring_info = (
            comm_ring_info.merge(comm_hash2id, on=["nodeId", "commHash"], how="left")
            .drop(columns=["commHash", "text", "start", "end", "eventId"])
            .drop_duplicates()
        )
    else:
        # Synthesize ring topology for NCCL 2.28
        # Create ring topology for multiple channels (up to 32, the NCCL default max)
        logger.warning("No ring topology markers found - synthesizing simple ring with 32 channels")
        num_channels = 32
        ring_rows = []
        for commId in comm_info["commId"].unique():
            comm_ranks = comm_info[comm_info["commId"] == commId].sort_values("rank")
            nRanks = len(comm_ranks)
            for channelId in range(num_channels):
                for _, row in comm_ranks.iterrows():
                    myRank = row["rank"]
                    prevRank = (myRank - 1) % nRanks
                    nextRank = (myRank + 1) % nRanks
                    ring_rows.append({
                        "commId": commId,
                        "channelId": channelId,
                        "prevRank": prevRank,
                        "myRank": myRank,
                        "nextRank": nextRank,
                        "pid": row["pid"],
                        "nodeId": row["nodeId"]
                    })
        if ring_rows:
            comm_ring_info = pd.DataFrame(ring_rows)
            comm_ring_info = convert_numeric(comm_ring_info, [], ["channelId", "prevRank", "myRank", "nextRank", "pid"])
        else:
            comm_ring_info = pd.DataFrame(columns=["commId", "channelId", "prevRank", "myRank", "nextRank", "pid", "nodeId"])

    logger.info("extracting communicator trees")
    comm_tree_pattern = r"commHash (0x[0-9a-f]+) Trees \[(\d+)\] (-?\d+)/(-?\d+)/(-?\d+)->(-?\d+)->(-?\d+) pid (\d+)"
    comm_tree_pattern_matching_indexs = data["text"].str.match(comm_tree_pattern)
    comm_tree_info = data[comm_tree_pattern_matching_indexs].copy()
    data = data[~comm_tree_pattern_matching_indexs]
    
    if len(comm_tree_info) > 0:
        comm_tree_info[
            [
                "commHash",
                "channelId",
                "child1Rank",
                "child2Rank",
                "child3Rank",
                "myRank",
                "parentRank",
                "pid",
            ]
        ] = comm_tree_info["text"].str.extract(comm_tree_pattern)
        comm_tree_info = convert_numeric(comm_tree_info,
            [
                "child1Rank",
                "child2Rank",
                "child3Rank",
                "parentRank"
            ],["channelId","myRank","pid"]
        )
        comm_tree_info = (
            comm_tree_info.merge(comm_hash2id, on=["nodeId", "commHash"], how="left")
            .drop(columns=["commHash", "text", "start", "end", "eventId"])
            .drop_duplicates()
        )
    else:
        # Synthesize tree topology for NCCL 2.28 (simple binary tree with 32 channels)
        logger.warning("No tree topology markers found - synthesizing simple tree with 32 channels")
        num_channels = 32
        tree_rows = []
        for commId in comm_info["commId"].unique():
            comm_ranks = comm_info[comm_info["commId"] == commId].sort_values("rank")
            nRanks = len(comm_ranks)
            for channelId in range(num_channels):
                for _, row in comm_ranks.iterrows():
                    myRank = row["rank"]
                    # Simple binary tree: parent is (rank-1)//2, children are 2*rank+1, 2*rank+2
                    parentRank = (myRank - 1) // 2 if myRank > 0 else -1
                    child1Rank = 2 * myRank + 1 if 2 * myRank + 1 < nRanks else -1
                    child2Rank = 2 * myRank + 2 if 2 * myRank + 2 < nRanks else -1
                    tree_rows.append({
                        "commId": commId,
                        "channelId": channelId,
                        "child1Rank": child1Rank,
                        "child2Rank": child2Rank,
                        "child3Rank": -1,
                        "myRank": myRank,
                        "parentRank": parentRank,
                        "pid": row["pid"],
                        "nodeId": row["nodeId"]
                    })
        if tree_rows:
            comm_tree_info = pd.DataFrame(tree_rows)
            comm_tree_info = convert_numeric(comm_tree_info,
                ["child1Rank", "child2Rank", "child3Rank", "parentRank"],
                ["channelId", "myRank", "pid"]
            )
        else:
            comm_tree_info = pd.DataFrame(columns=["commId", "channelId", "child1Rank", "child2Rank", "child3Rank", "myRank", "parentRank", "pid", "nodeId"])
    
    return comm_info, comm_ring_info, comm_tree_info, data


def get_profiling_interval(_data: pd.DataFrame) -> pd.DataFrame:
    logger.info("extracting profiling intervals")
    data = _data[["nodeId", "start", "text"]].copy()
    profile_start_pattern = r"nsys profiling start, pid: (\d+)"
    profile_end_pattern = r"nsys profiling stopped, pid: (\d+)"
    profile_start_pattern_matching_indexs = data["text"].str.match(profile_start_pattern)
    profile_end_pattern_matching_indexs = data["text"].str.match(profile_end_pattern)
    profile_start_info = data[profile_start_pattern_matching_indexs].copy()
    profile_end_info = data[profile_end_pattern_matching_indexs].copy()
    _data = _data[~(profile_start_pattern_matching_indexs | profile_end_pattern_matching_indexs)]
    profile_start_info["pid"] = (
        profile_start_info["text"].str.extract(profile_start_pattern).astype("Int64")
    )
    profile_start_info = profile_start_info.drop(columns=["text"])
    profile_end_info["pid"] = (
        profile_end_info["text"].str.extract(profile_end_pattern).astype("Int64")
    )
    profile_end_info = profile_end_info.rename(columns={"start": "end"}).drop(
        columns=["text"]
    )
    result_df = profile_start_info.merge(profile_end_info, on=["nodeId", "pid"])[
        ["nodeId", "pid", "start", "end"]
    ]
    return {(row["nodeId"], row["pid"]): (row["start"], row["end"]) for _, row in result_df.iterrows()} ,_data


@numba.njit
def _associate_events(interval_starts, interval_ends, interval_id, events_time):
    associated_ids = -1 * np.ones(len(events_time), dtype=np.int64)
    j = 0
    for i in range(len(events_time)):
        while j < len(interval_starts) and interval_ends[j] < events_time[i]:
            j += 1
        if (
            j < len(interval_starts)
            and interval_starts[j] <= events_time[i] < interval_ends[j]
        ):
            associated_ids[i] = interval_id[j]
    return associated_ids


def filter_time(profiling_interval: Dict[Tuple[int, int], Tuple[int, int]], data: Dict[Tuple[int, int], pd.DataFrame]):
    result_dfs = {}
    logger.info("filtering events by profiling intervals")
    for gpu, gpu_df in tqdm(data.items(), total=len(data)):
        if gpu not in profiling_interval:
            logger.warning(f"GPU {gpu} has no profiling interval, skipping filtering")
            continue
        profile_start, profile_end = profiling_interval[gpu]
        result_dfs[gpu] = gpu_df[
            (gpu_df["start"] < profile_end) & (gpu_df["end"] > profile_start)
        ].reset_index(drop=True)
    return result_dfs


@numba.njit
def _associate_start_ends(sequence, internal_groups=True):
    group_id = -1
    group_ids = -1 * np.ones(len(sequence), dtype=np.int64)
    stack = []
    for i in range(len(sequence)):
        if sequence[i]:
            curr_group_id = -1
            if internal_groups or len(stack) == 0:
                curr_group_id = group_id + 1
                group_id += 1
            group_ids[i] = curr_group_id
            stack.append(curr_group_id)
        else:
            group_ids[i] = stack.pop()
    assert len(stack) == 0, "Mismatched start and end events"
    return group_ids


def get_event_info(data: pd.DataFrame, comm_info: pd.DataFrame = None):
    logger.info("extracting event infos")
    logger.info("extracting kernel group events")
    kernel_group_start_pattern = r"ncclGroupStart\(\): pid (\d+)"
    kernel_group_end_pattern = r"ncclGroupEnd\(\): pid (\d+)"
    kernel_group_start_pattern_matching_indexs = data["text"].str.match(kernel_group_start_pattern)
    kernel_group_end_pattern_matching_indexs = data["text"].str.match(kernel_group_end_pattern)
    kernel_group_start_info = data[
        kernel_group_start_pattern_matching_indexs
    ].copy()
    kernel_group_end_info = data[
        kernel_group_end_pattern_matching_indexs
    ].copy()
    data = data[~(kernel_group_start_pattern_matching_indexs | kernel_group_end_pattern_matching_indexs)]
    kernel_group_start_info["pid"] = (
        kernel_group_start_info["text"]
        .str.extract(kernel_group_start_pattern)
        .astype("Int64")
    )
    kernel_group_end_info["pid"] = (
        kernel_group_end_info["text"]
        .str.extract(kernel_group_end_pattern)
        .astype("Int64")
    )

    kernel_group_start_info.drop(columns=["text"], inplace=True)
    kernel_group_end_info.drop(columns=["text"], inplace=True)
    kernel_group_start_info["isStart"] = True
    kernel_group_end_info["isStart"] = False
    # concat start and end info
    kernel_group_start_end = (
        pd.concat([kernel_group_start_info, kernel_group_end_info], ignore_index=True)
        .sort_values(by=["nodeId", "pid", "start"])
        .reset_index(drop=True)
    )
    group_start_end_grouped = {
        name: group for name, group in kernel_group_start_end.groupby(["nodeId", "pid"])
    }

    for gpu, group in group_start_end_grouped.items():
        group = group.sort_values(by="start").reset_index(drop=True)
        group["groupId"] = _associate_start_ends(group["isStart"].to_numpy(), False)
        group_starts = group[group["isStart"]].rename(columns={"start": "group_start"})[
            ["groupId", "group_start"]
        ]
        group_ends = group[~group["isStart"]].rename(columns={"start": "group_end"})[
            ["groupId", "group_end"]
        ]
        group = group_starts.merge(group_ends, on=["groupId"], how="inner")
        group = group[group["groupId"] != -1]
        group_start_end_grouped[gpu] = group
    
    logger.info("extracting collective kernel events")
    coll_kernel_pattern = r"nWarps \d+ count (\d+) chunkCount (\d+) workCount (\d+) lastChunkCount (\d+) workOffset (\d+) sendbuff (\d+) recvbuff (\d+) pid (\d+)"
    coll_kernel_pattern_matching_indexs = data["text"].str.match(coll_kernel_pattern)
    coll_kernel_data = data[coll_kernel_pattern_matching_indexs].copy()
    data = data[~coll_kernel_pattern_matching_indexs]
    coll_kernel_data[
        [
            # "nWarps",
            "count",
            "chunkCount",
            "workCount",
            "lastChunkCount",
            "workOffset",
            "sendbuff",
            "recvbuff",
            "pid",
        ]
    ] = coll_kernel_data["text"].str.extract(coll_kernel_pattern)
    coll_kernel_data.drop(columns=["text"], inplace=True)
    coll_kernel_data = convert_numeric(coll_kernel_data,
        [],[
            # "nWarps",
            "count",
            "chunkCount",
            "workCount",
            "lastChunkCount",
            "workOffset",
            "sendbuff",
            "recvbuff",
            "pid",
        ]
    )
    
    logger.info("extracting P2P kernel events")
    p2p_kernel_pattern = r"Bytes (\d+) nWarps \d+ p2pType (\d+) peer (\d+) proto (\d+) countHi32 (\d+) countLo32 (\d+) chunkSize (\d+) pid (\d+)"
    p2p_kernel_pattern_matching_indexs = data["text"].str.match(p2p_kernel_pattern)
    p2p_kernel_data = data[p2p_kernel_pattern_matching_indexs].copy()
    data = data[~p2p_kernel_pattern_matching_indexs]
    p2p_kernel_data[
        [
            "Bytes",
            # "nWarps",
            "p2pType",
            "peer",
            "proto",
            "countHi32",
            "countLo32",
            "chunkSize",
            "pid",
        ]
    ] = p2p_kernel_data["text"].str.extract(p2p_kernel_pattern)
    p2p_kernel_data.drop(columns=["text"], inplace=True)
    p2p_kernel_data = convert_numeric(p2p_kernel_data,
        [],[
            "Bytes",
            # "nWarps",
            "peer",
            "proto",
            "countHi32",
            "countLo32",
            "chunkSize",
            "pid",
        ]
    )
    
    logger.info("extracting communication events")
    comm_pattern = r"nccl([a-zA-Z]+)\(\): commHash (0x[0-9a-f]+), stream (0x[0-9a-f]+), data_size \d+, type_size \d+,.* pid (\d+)"
    comm_pattern_matching_indexs = data["text"].str.match(comm_pattern)
    comm_data = data[comm_pattern_matching_indexs].copy()
    data = data[~comm_pattern_matching_indexs]
    comm_data[["collective", "commHash", "stream", "pid"]] = (
        comm_data["text"].str.extract(comm_pattern)
    )
    comm_data.drop(columns=["text"], inplace=True)
    comm_data["pid"] = comm_data["pid"].astype("Int64")
    if comm_info is not None:
        comm_data = comm_data.merge(
            comm_info[["nodeId", "commHash", "commId"]].drop_duplicates(), on=["nodeId", "commHash"], how="left"
        )
        
    logger.info("extracting collective info events")
    coll_info_pattern = r"collType (\d+) root (\d+) redOp (\d+) algo (\d+) proto (\d+) commHash (\S+) stream (\S+) data_size (\d+) type_size (\d+) chunkSize \d+ chunkCount \d+ chunkSteps (\d+) sliceSteps (\d+) stepSize (\d+) pid (\d+)"
    coll_info_pattern_matching_indexs = data["text"].str.match(coll_info_pattern)
    coll_info_data = data[coll_info_pattern_matching_indexs].copy()
    data = data[~coll_info_pattern_matching_indexs]
    coll_info_data[
        [
            "collType",
            "root",
            "redOp",
            "algo",
            "proto",
            "commHash",
            "stream",
            "data_size",
            "type_size",
            # "chunkSize",
            # "chunkCount",
            "chunkSteps",
            "sliceSteps",
            "stepSize",
            "pid",
        ]
    ] = coll_info_data["text"].str.extract(coll_info_pattern)
    coll_info_data.drop(columns=["text"], inplace=True)
    coll_info_data = convert_numeric(coll_info_data,
        [],[
            "collType",
            "root",
            "redOp",
            "algo",
            "proto",
            "data_size",
            "type_size",
            # "chunkSize",
            # "chunkCount",
            "chunkSteps",
            "sliceSteps",
            "stepSize",
            "pid",
        ]
    )

    logger.info("grouping events by GPU")
    comm_grouped = {name: group for name, group in comm_data.groupby(["nodeId", "pid"])}
    coll_info_grouped = {
        name: group for name, group in coll_info_data.groupby(["nodeId", "pid"])
    }
    coll_kernel_grouped = {
        name: group for name, group in coll_kernel_data.groupby(["nodeId", "pid"])
    }
    p2p_kernel_grouped = {
        name: group for name, group in p2p_kernel_data.groupby(["nodeId", "pid"])
    }

    logger.info("associating events")
    for gpu in tqdm(comm_grouped.keys()):
        comm = comm_grouped[gpu]
        comm = comm.sort_values(by="start").reset_index(drop=True)
        group_start_ends = group_start_end_grouped[gpu]
        comm["groupId"] = _associate_events(
            group_start_ends["group_start"].to_numpy(),
            group_start_ends["group_end"].to_numpy(),
            group_start_ends["groupId"].to_numpy(),
            comm["start"].to_numpy(),
        )
        comm_grouped[gpu] = comm
        coll_comm = comm[
            (comm["collective"] != "Send") & (comm["collective"] != "Recv")
        ]
        if len(coll_comm) > 0:
            # # From https://github.com/NVIDIA/nccl/blob/81c7da31f98f4aa4a70317169ba8ca130839b447/src/include/nccl_common.h#L59
            # coll_types = {
            #     "Broadcast": "0",
            #     "Reduce": "1",
            #     "AllGather": "2",
            #     "ReduceScatter": "3",
            #     "AllReduce": "4",
            #     "AllToAll": "8",
            #     "Scatter": "9",
            #     "Gather": "10",
            #     "AllGatherV": "11",
            # }
            coll_infos = coll_info_grouped[gpu]
            coll_kernels = coll_kernel_grouped[gpu]

            coll_infos = coll_infos.sort_values(by="start").reset_index(drop=True)
            coll_kernels = coll_kernels.sort_values(by="start").reset_index(drop=True)

            comm_starts = coll_comm["start"].to_numpy()
            comm_ends = np.concat([comm_starts[1:], np.array([np.iinfo(np.int64).max])])

            coll_info_starts = coll_infos["start"].to_numpy()
            coll_infos["association"] = _associate_events(
                comm_starts,
                comm_ends,
                coll_comm["eventId"].to_numpy(),
                coll_info_starts,
            )

            coll_kernel_starts = coll_kernels["start"].to_numpy()
            coll_kernels["association"] = _associate_events(
                comm_starts,
                comm_ends,
                coll_comm["eventId"].to_numpy(),
                coll_kernel_starts,
            )
            coll_info_grouped[gpu] = coll_infos
            coll_kernel_grouped[gpu] = coll_kernels

        p2p_comm = comm[(comm["collective"] == "Send") | (comm["collective"] == "Recv")]
        if gpu in p2p_kernel_grouped:
            p2p_kernels = p2p_kernel_grouped[gpu]
        else:
            p2p_kernels = pd.DataFrame()
        if len(p2p_comm) > 0 and len(p2p_kernels) > 0:
            p2p_types = {
                "Send": "1",
                "Recv": "2",
            }
            p2p_kernel_lists = []
            for p2p_type, type_id in p2p_types.items():
                curr_p2p_comm = p2p_comm[p2p_comm["collective"] == p2p_type].sort_values(by="start").reset_index(drop=True)
                curr_p2p_kernels = p2p_kernels[p2p_kernels["p2pType"] == type_id].sort_values(by="start").reset_index(drop=True).copy()
                comm_starts = curr_p2p_comm["start"].to_numpy()
                comm_ends = np.concat([comm_starts[1:], np.array([np.iinfo(np.int64).max])])
                p2p_kernel_starts = curr_p2p_kernels["start"].to_numpy()
                curr_p2p_kernels["association"] = _associate_events(
                    comm_starts,
                    comm_ends,
                    curr_p2p_comm["eventId"].to_numpy(),
                    p2p_kernel_starts,
                )
                p2p_kernel_lists.append(curr_p2p_kernels)
            p2p_kernels = pd.concat(p2p_kernel_lists, ignore_index=True)
            p2p_kernel_grouped[gpu] = p2p_kernels
    
    comm_grouped = {k: v.drop(columns=["commHash", "nodeId", "pid"]) for k, v in comm_grouped.items()}
    coll_info_grouped = {k: v.drop(columns=["eventId", "start", "end", "commHash", "stream", "nodeId", "pid"]) for k, v in coll_info_grouped.items()}
    coll_kernel_grouped = {k: v.drop(columns=["eventId", "start", "end", "nodeId", "pid"]) for k, v in coll_kernel_grouped.items()}
    p2p_kernel_grouped = {k: v.drop(columns=["eventId", "start", "end", "nodeId", "pid"]) for k, v in p2p_kernel_grouped.items()}
    return comm_grouped, coll_info_grouped, coll_kernel_grouped, p2p_kernel_grouped, data


def associate_kernel_to_nvtx(
    comm_grouped: pd.DataFrame,
    kernel_events: pd.DataFrame,
    profiling_interval: pd.DataFrame = None,
):
    """Associate kernel events to NVTX events using time-based matching.
    
    For NCCL 2.28+ traces, kernel events run on many CUDA streams while NVTX
    events only track a few streams. We use time overlap to associate kernels
    to their corresponding NVTX collective calls.
    """
    if profiling_interval is not None:
        logger.info("filtering kernel events by profiling intervals")
        kernel_events = filter_time(profiling_interval, kernel_events)
    logger.info("associating kernels to nvtx events (time-based matching)")
    kernel_df_grouped = {
        name: group for name, group in kernel_events.groupby(["nodeId", "pid"])
    }
    
    for gpu in tqdm(kernel_df_grouped.keys()):
        kernels = kernel_df_grouped[gpu].sort_values(by="start").reset_index(drop=True)
        nvtxs = comm_grouped[gpu].sort_values(by="start").reset_index(drop=True)
        
        # Handle grouped vs non-grouped NVTX events
        non_grouped_nvtxs = nvtxs[nvtxs["groupId"] == -1]
        grouped_nvtxs = nvtxs[nvtxs["groupId"] != -1]

        first_nvtxs_in_group = grouped_nvtxs.groupby("groupId").first().reset_index()
        dropped_nvtxs = (
            grouped_nvtxs.groupby("groupId")
            .apply(lambda x: x.iloc[1:], include_groups=False)
            .reset_index(level=0)
            .reset_index(drop=True)
        )

        nvtxs = (
            pd.concat([non_grouped_nvtxs, first_nvtxs_in_group], ignore_index=True)
            .sort_values(by="start")
            .reset_index(drop=True)
        )
        
        # Time-based association: assign each kernel to the NVTX event that contains it
        # Use the NVTX start/end times to create intervals
        nvtx_starts = nvtxs["start"].to_numpy()
        nvtx_ends = nvtxs["end"].to_numpy()
        nvtx_event_ids = nvtxs["eventId"].to_numpy()
        kernel_starts = kernels["start"].to_numpy()
        
        # Associate kernels to NVTX events by finding which NVTX interval contains each kernel
        kernel_associations = _associate_events(
            nvtx_starts,
            nvtx_ends,
            nvtx_event_ids,
            kernel_starts,
        )
        kernels["association"] = kernel_associations
        
        # For kernels that couldn't be associated (outside any NVTX interval),
        # try to associate by finding the closest preceding NVTX event
        unassociated_mask = kernels["association"] == -1
        if unassociated_mask.any():
            unassociated_times = kernels.loc[unassociated_mask, "start"].to_numpy()
            # Find the closest preceding NVTX event
            associations = np.searchsorted(nvtx_starts, unassociated_times, side='right') - 1
            associations = np.clip(associations, 0, len(nvtx_event_ids) - 1)
            kernels.loc[unassociated_mask, "association"] = nvtx_event_ids[associations]

        kernel_df_grouped[gpu] = kernels
        
        # Update NVTX events with kernel timing info
        # Use first and last kernel times for each NVTX event
        kernel_times = kernels.groupby("association").agg(
            kernel_start=("start", "min"),
            kernel_end=("end", "max")
        ).reset_index().rename(columns={"association": "eventId"})
        
        nvtxs = nvtxs.merge(kernel_times, on="eventId", how="left")
        # Use kernel times if available, otherwise keep original NVTX times
        nvtxs["start"] = nvtxs["kernel_start"].fillna(nvtxs["start"]).astype("Int64")
        nvtxs["end"] = nvtxs["kernel_end"].fillna(nvtxs["end"]).astype("Int64")
        nvtxs = nvtxs.drop(columns=["kernel_start", "kernel_end"])

        # Handle dropped NVTX events (grouped events after the first)
        if len(dropped_nvtxs) > 0:
            dropped_nvtxs = dropped_nvtxs.drop(columns=["start", "end"], errors='ignore')
            # Get end time from the group's first NVTX
            group_ends = nvtxs[["groupId", "end"]].drop_duplicates()
            dropped_nvtxs = dropped_nvtxs.merge(group_ends, on="groupId", how="left")
            dropped_nvtxs["start"] = dropped_nvtxs["end"]
            
            nvtxs = (
                pd.concat([nvtxs, dropped_nvtxs], ignore_index=True)
                .sort_values(by=["start"])
                .reset_index(drop=True)
            )

        comm_grouped[gpu] = nvtxs

    return comm_grouped
    
def add_context_parallelism(comm_datas: Dict[Tuple[int, int], pd.DataFrame]):
    logger.info("adding context parallelism information")
    collective_labels = {
        "AllGather": "A",
        "AllReduce": "B",
        "Broadcast": "C",
        "ReduceScatter": "D",
        "Recv": "E",
        "Send": "F"
    }

    rules = []
    def register_rule(parallelism_name):
        def decorator(func):
            rules.append((parallelism_name, func))
            return func
        return decorator
    
    def get_rule(label_seq):
        for parallelism_name, rule_func in rules:
            if rule_func(label_seq):
                return parallelism_name
        return "Other"

    @register_rule("DP")
    def rule_dp(label_seq):
        # for fully sharded data parallelism, the collective sequence should be like AAADDDAAADDD...
        # get the number of starting As
        n_a = 0
        for c in label_seq:
            if c == "A":
                n_a += 1
            else:
                break
        if n_a == 0:
            return False
        # check if the rest of the sequence is made of D and A in alternating blocks of size n_a
        for i in range(0, len(label_seq), 2 * n_a):
            if label_seq[i:i+n_a] != "A" * n_a:
                return False
            if label_seq[i+n_a:i+2*n_a] != "D" * n_a and i + n_a < len(label_seq):
                return False
        return True
    
    @register_rule("PP")
    def rule_pp(label_seq):
        # for pipeline parallelism, it should just be a sequence of alternating E and F
        # the length of the sequence should be even
        if len(label_seq) % 2 != 0:
            return False
        first_two = label_seq[:2]
        if first_two not in ["EF", "FE"]:
            return False
        for i in range(0, len(label_seq), 2):
            if label_seq[i : i + 2] != first_two:
                return False
        return True
    
    @register_rule("PP")
    def rule_pp2(label_seq):
        # for pipeline parallelism variant, it should just be a sequence of alternating E and F with possible multiple Es or Fs in a row
        # e.g., EEEFFFEFEEFF
        label_seq = "".join(c for c in label_seq if c in ["E", "F"])
        if len(label_seq) < 2:
            return False
        first_char = label_seq[0]
        second_char = "F" if first_char == "E" else "E"
        expecting_first = True
        for c in label_seq:
            if expecting_first:
                if c != first_char:
                    expecting_first = False
            else:
                if c != second_char:
                    expecting_first = True
        return True
    
    for gpu, comm_data in tqdm(comm_datas.items(), total=len(comm_datas)):
        comm_data = comm_data.sort_values("start").reset_index(drop=True)
        comm_data["label"] = comm_data["collective"].map(collective_labels)
        comm_grouped = (
            comm_data.groupby(["stream"])
            .agg(label_seq=("label", lambda x: "".join(x)))
            .reset_index()
        )
        comm_grouped["parallelism"] = comm_grouped["label_seq"].map(get_rule)
        comm_datas[gpu] = comm_data.merge(
            comm_grouped[["stream", "parallelism"]],
            on=["stream"],
            how="left",
        ).drop(columns=["label"])
    return comm_datas



if __name__ == "__main__":
    traces = find_all_traces("traces/Llama70B_N64_GPU256_TP1_PP8_DP32_70B_BS32/sqlite")
    kernel_events = get_kernel_events(traces)
    nvtx_events = get_nvtx_events(traces)
    comm_info, comm_ring_info, comm_tree_info = get_communicator_info(nvtx_events)
    profiling_interval = get_profiling_interval(nvtx_events)
    comm_data, coll_info, coll_kernels, p2p_kernels = get_event_info(
        nvtx_events, profiling_interval
    )
    kernel_events = associate_kernel_to_nvtx(
        comm_data, kernel_events, profiling_interval
    )
    comm_data = add_context_parallelism(comm_data)
