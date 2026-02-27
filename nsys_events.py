"""
Sequential implementation of NSYS event processing.

This module processes trace files sequentially and is simpler but slower
than the Dask-based parallel implementation.
"""

import logging
import os
import re
import sqlite3
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Import shared utilities from common module
from nsys_events_common import (
    # Core utilities
    find_all_traces,
    convert_numeric,
    logger,
    # Patterns and schemas
    NVTX_PATTERNS,
    # Numba functions
    _associate_events,
    _associate_start_ends,
    # Data processing
    filter_time,
    filter_time_single,
    add_context_parallelism,
    # Kernel association
    process_one_gpu_kernels,
    COLLECTIVE_LABELS,
    COLLECTIVE_LABELS_KERNEL,
)


def get_kernel_events(traces: List[os.PathLike]) -> pd.DataFrame:
    """Read kernel events from all trace files sequentially."""
    logger.info("querying for kernel events")
    kernel_dfs = []
    for trace_file in tqdm(traces):
        node_id = re.search(r"nid(\d+)", trace_file.name).group(1)
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
    return kernel_df


def get_nvtx_events(traces: List[os.PathLike]) -> pd.DataFrame:
    """Read NVTX events from all trace files sequentially."""
    logger.info("querying for nvtx events")
    nvtx_dfs = []
    for trace_file in tqdm(traces):
        node_id = re.search(r"nid(\d+)", trace_file.name).group(1)
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


def get_communicator_info(data: pd.DataFrame):
    """Extract communicator info from NVTX events."""
    logger.info("extracting communicator info")
    # extract available informations from the table
    comm_info_pattern = NVTX_PATTERNS["comm_info"]
    comm_info_pattern_matching_indexs = data["text"].str.match(comm_info_pattern)
    comm_info = data[comm_info_pattern_matching_indexs].copy()
    data = data[~comm_info_pattern_matching_indexs]
    comm_info[["commHash", "commId", "rank", "nRanks", "pid"]] = comm_info[
        "text"
    ].str.extract(comm_info_pattern)
    comm_info = convert_numeric(comm_info, [], ["rank", "nRanks", "pid"])
    comm_info = comm_info.drop(
        columns=["text", "start", "end", "eventId"]
    ).drop_duplicates()

    comm_hash2id = comm_info[["nodeId", "commHash", "commId"]].drop_duplicates()

    logger.info("extracting communicator rings")
    comm_ring_pattern = NVTX_PATTERNS["comm_ring"]
    comm_ring_pattern_matching_indexs = data["text"].str.match(comm_ring_pattern)
    comm_ring_info = data[comm_ring_pattern_matching_indexs].copy()
    data = data[~comm_ring_pattern_matching_indexs]
    comm_ring_info[
        ["commHash", "channelId", "prevRank", "myRank", "nextRank", "pid"]
    ] = comm_ring_info["text"].str.extract(comm_ring_pattern)
    comm_ring_info = convert_numeric(comm_ring_info, [], ["channelId", "prevRank", "myRank", "nextRank", "pid"])
    comm_ring_info = (
        comm_ring_info.merge(comm_hash2id, on=["nodeId", "commHash"], how="left")
        .drop(columns=["commHash", "text", "start", "end", "eventId"])
        .drop_duplicates()
    )

    logger.info("extracting communicator trees")
    comm_tree_pattern = NVTX_PATTERNS["comm_tree"]
    comm_tree_pattern_matching_indexs = data["text"].str.match(comm_tree_pattern)
    comm_tree_info = data[comm_tree_pattern_matching_indexs].copy()
    data = data[~comm_tree_pattern_matching_indexs]
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
    return comm_info, comm_ring_info, comm_tree_info, data


def get_profiling_interval(_data: pd.DataFrame) -> pd.DataFrame:
    """Extract profiling intervals from NVTX events."""
    logger.info("extracting profiling intervals")
    data = _data[["nodeId", "start", "text"]].copy()
    profile_start_pattern = NVTX_PATTERNS["profile_start"]
    profile_end_pattern = NVTX_PATTERNS["profile_end"]
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
    return {(row["nodeId"], row["pid"]): (row["start"], row["end"]) for _, row in result_df.iterrows()}, _data


def get_event_info(data: pd.DataFrame, comm_info: pd.DataFrame = None):
    """Extract and associate communication events from NVTX data."""
    logger.info("extracting event infos")
    logger.info("extracting kernel group events")
    kernel_group_start_pattern = NVTX_PATTERNS["group_start"]
    kernel_group_end_pattern = NVTX_PATTERNS["group_end"]
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
    coll_kernel_pattern = NVTX_PATTERNS["coll_kernel"]
    coll_kernel_pattern_matching_indexs = data["text"].str.match(coll_kernel_pattern)
    coll_kernel_data = data[coll_kernel_pattern_matching_indexs].copy()
    data = data[~coll_kernel_pattern_matching_indexs]
    coll_kernel_data[
        [
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
    p2p_kernel_pattern = NVTX_PATTERNS["p2p_kernel"]
    p2p_kernel_pattern_matching_indexs = data["text"].str.match(p2p_kernel_pattern)
    p2p_kernel_data = data[p2p_kernel_pattern_matching_indexs].copy()
    data = data[~p2p_kernel_pattern_matching_indexs]
    p2p_kernel_data[
        [
            "Bytes",
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
            "peer",
            "proto",
            "countHi32",
            "countLo32",
            "chunkSize",
            "pid",
        ]
    )
    
    logger.info("extracting communication events")
    comm_pattern = NVTX_PATTERNS["comm"]
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
    coll_info_pattern = NVTX_PATTERNS["coll_info"]
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
            "chunkSteps",
            "sliceSteps",
            "stepSize",
            "pid",
        ]
    )

    logger.info("grouping events by GPU")
    comm_grouped = defaultdict(
        lambda: pd.DataFrame(columns=comm_data.columns),
        {name: group for name, group in comm_data.groupby(["nodeId", "pid"])}
    )
    coll_info_grouped = defaultdict(
        lambda: pd.DataFrame(columns=list(coll_info_data.columns) + ["association"]),
        {name: group for name, group in coll_info_data.groupby(["nodeId", "pid"])}
    )
    coll_kernel_grouped = defaultdict(
        lambda: pd.DataFrame(columns=list(coll_kernel_data.columns) + ["association"]),
        {name: group for name, group in coll_kernel_data.groupby(["nodeId", "pid"])}
    )
    p2p_kernel_grouped = defaultdict(
        lambda: pd.DataFrame(columns=list(p2p_kernel_data.columns) + ["association"]),
        {name: group for name, group in p2p_kernel_data.groupby(["nodeId", "pid"])}
    )

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
        p2p_kernels = p2p_kernel_grouped[gpu]
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
    profiling_interval: pd.DataFrame = dict(),
):
    """Associate kernel events to NVTX events (sequential implementation)."""

    logger.info("associating kernels to nvtx events")
    kernel_df_grouped = {
        name: group for name, group in kernel_events.groupby(["nodeId", "pid"])
    }
    
    for gpu in tqdm(kernel_df_grouped.keys()):
        kernels = kernel_df_grouped[gpu]
        nvtxs = comm_grouped[gpu]
        interval = profiling_interval.get(gpu, None)
        
        # Use shared function for kernel association
        _, kernel_times = process_one_gpu_kernels(gpu, kernels, nvtxs, interval)
        
        # Merge kernel times back - rebuild nvtxs with new times
        nvtxs = nvtxs.sort_values(by="start").reset_index(drop=True)
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
        
        # Drop old times and merge new ones
        nvtxs = nvtxs.drop(columns=["start", "end"]).merge(
            kernel_times, on="eventId", how="left"
        )
        
        # Handle dropped nvtxs
        if len(dropped_nvtxs) > 0:
            dropped_nvtxs = dropped_nvtxs.drop(columns=["start", "end"]).merge(
                nvtxs[["end", "groupId"]].drop_duplicates(), on=["groupId"], how="left"
            )
            dropped_nvtxs["start"] = dropped_nvtxs["end"]
            nvtxs = (
                pd.concat([nvtxs, dropped_nvtxs], ignore_index=True)
                .sort_values(by=["start"])
                .reset_index(drop=True)
            )
        
        comm_grouped[gpu] = nvtxs

    return comm_grouped


if __name__ == "__main__":
    traces = find_all_traces("traces/Llama70B_N64_GPU256_TP1_PP8_DP32_70B_BS32/sqlite")
    kernel_events = get_kernel_events(traces)
    nvtx_events = get_nvtx_events(traces)
    comm_info, comm_ring_info, comm_tree_info, nvtx_events = get_communicator_info(nvtx_events)
    profiling_interval, nvtx_events = get_profiling_interval(nvtx_events)
    comm_data, coll_info, coll_kernels, p2p_kernels, nvtx_events = get_event_info(
        nvtx_events, comm_info
    )
    kernel_events = associate_kernel_to_nvtx(
        comm_data, kernel_events, profiling_interval
    )
    comm_data = add_context_parallelism(comm_data)
