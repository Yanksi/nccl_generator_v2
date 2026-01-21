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


def get_kernel_events(traces: List[os.PathLike]) -> pd.DataFrame:
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

    kernel_df = pd.concat(kernel_dfs, ignore_index=True)
    kernel_df.drop(columns=["value"], inplace=True)
    # kernel_df['eventId'] = range(len(kernel_df))  # Add unique ID to each row
    # kernel_df["stream"] = kernel_df[["deviceId", "streamId"]].apply(lambda x: f"{x['deviceId']}-{x['streamId']}", axis=1)
    return kernel_df


def get_nvtx_events(traces: List[os.PathLike]) -> pd.DataFrame:
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

    nvtx_df = pd.concat(nvtx_dfs, ignore_index=True)
    nvtx_df["eventId"] = range(len(nvtx_df))  # Add unique ID to each row
    return nvtx_df


def convert_numeric(data_frame: pd.DataFrame, columns: List[str], safe: bool = True) -> pd.DataFrame:
    if safe:
        int64_max = np.iinfo(np.int64).max
        int64_min = np.iinfo(np.int64).min
        data_frame[columns] = (
            data_frame[columns]
            .apply(pd.to_numeric, errors="coerce")
            .clip(int64_min, int64_max)
            .astype("Int64")
        )
    else:
        data_frame[columns] = data_frame[columns].astype("Int64")
    return data_frame


def get_communicator_info(data: pd.DataFrame):
    logger.info("extracting communicator info")
    # extract available informations from the table
    comm_info_pattern = (
        r"commHash (0x[0-9a-f]+) commId (0x[0-9a-f]+) rank (\d+) nranks (\d+) pid (\d+)"
    )
    comm_info = data[data["text"].str.match(comm_info_pattern)].copy()
    comm_info[["commHash", "commId", "rank", "nRanks", "pid"]] = comm_info[
        "text"
    ].str.extract(comm_info_pattern)
    comm_info = convert_numeric(comm_info, ["rank", "nRanks", "pid"], False)
    # comm_info[["nRanks", "rank", "pid"]] = comm_info[["nRanks", "rank", "pid"]].astype(
    #     "Int64"
    # )
    comm_info = comm_info.drop(
        columns=["text", "start", "end", "eventId"]
    ).drop_duplicates()

    comm_hash2id = comm_info[["nodeId", "commHash", "commId"]].drop_duplicates()

    logger.info("extracting communicator rings")
    comm_ring_pattern = (
        r"commHash (0x[0-9a-f]+) Rings \[(\d+)\] (\d+)->(\d+)->(\d+) pid (\d+)"
    )
    comm_ring_info = data[data["text"].str.match(comm_ring_pattern)].copy()
    comm_ring_info[
        ["commHash", "channelId", "prevRank", "myRank", "nextRank", "pid"]
    ] = comm_ring_info["text"].str.extract(comm_ring_pattern)
    comm_ring_info = convert_numeric(comm_ring_info, ["channelId", "prevRank", "myRank", "nextRank", "pid"], False)
    comm_ring_info = (
        comm_ring_info.merge(comm_hash2id, on=["nodeId", "commHash"], how="left")
        .drop(columns=["commHash", "text", "start", "end", "eventId"])
        .drop_duplicates()
    )

    logger.info("extracting communicator trees")
    comm_tree_pattern = r"commHash (0x[0-9a-f]+) Trees \[(\d+)\] (-?\d+)/(-?\d+)/(-?\d+)->(-?\d+)->(-?\d+) pid (\d+)"
    comm_tree_info = data[data["text"].str.match(comm_tree_pattern)].copy()
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
            "channelId",
            "child1Rank",
            "child2Rank",
            "child3Rank",
            "myRank",
            "parentRank",
            "pid",
        ], False
    )
    comm_tree_info = (
        comm_tree_info.merge(comm_hash2id, on=["nodeId", "commHash"], how="left")
        .drop(columns=["commHash", "text", "start", "end", "eventId"])
        .drop_duplicates()
    )
    return comm_info, comm_ring_info, comm_tree_info


def get_profiling_interval(data: pd.DataFrame) -> pd.DataFrame:
    logger.info("extracting profiling intervals")
    data = data[["nodeId", "start", "text"]].copy()
    profile_start_pattern = r"nsys profiling start, pid: (\d+)"
    profile_end_pattern = r"nsys profiling stopped, pid: (\d+)"
    profile_start_info = data[data["text"].str.match(profile_start_pattern)].copy()
    profile_end_info = data[data["text"].str.match(profile_end_pattern)].copy()
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
    return profile_start_info.merge(profile_end_info, on=["nodeId", "pid"])[
        ["nodeId", "pid", "start", "end"]
    ]


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


def filter_time(profiling_interval: pd.DataFrame, data: pd.DataFrame):
    merged = data.merge(
        profiling_interval, on=["nodeId", "pid"], suffixes=("", "_profile")
    )
    return merged[
        (merged["start"] < merged["end_profile"])
        & (merged["end"] > merged["start_profile"])
    ].drop(columns=["start_profile", "end_profile"])


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


def get_event_info(data: pd.DataFrame, comm_info: pd.DataFrame = None, profiling_interval: pd.DataFrame = None):
    logger.info("extracting event infos")
    
    logger.info("extracting collective kernel events")
    coll_kernel_pattern = r"nWarps (\d+) count (\d+) chunkCount (\d+) workCount (\d+) lastChunkCount (\d+) workOffset (\d+) sendbuff (\d+) recvbuff (\d+) pid (\d+)"
    coll_kernel_pattern_matching_indexs = data["text"].str.match(coll_kernel_pattern)
    coll_kernel_data = data[coll_kernel_pattern_matching_indexs].copy()
    data = data[~coll_kernel_pattern_matching_indexs]
    coll_kernel_data[
        [
            "nWarps",
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
        [
            "nWarps",
            "count",
            "chunkCount",
            "workCount",
            "workOffset",
            "sendbuff",
            "recvbuff",
            "pid",
        ],
        False
    )
    coll_kernel_data = convert_numeric(coll_kernel_data, ["lastChunkCount"], True)
    
    logger.info("extracting P2P kernel events")
    p2p_kernel_pattern = r"Bytes (\d+) nWarps (\d+) p2pType (\d+) peer (\d+) proto (\d+) countHi32 (\d+) countLo32 (\d+) chunkSize (\d+) pid (\d+)"
    p2p_kernel_pattern_matching_indexs = data["text"].str.match(p2p_kernel_pattern)
    p2p_kernel_data = data[p2p_kernel_pattern_matching_indexs].copy()
    data = data[~p2p_kernel_pattern_matching_indexs]
    p2p_kernel_data[
        [
            "Bytes",
            "nWarps",
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
        [
            "Bytes",
            "nWarps",
            "peer",
            "proto",
            "countHi32",
            "countLo32",
            "chunkSize",
            "pid",
        ],
        False
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
        [
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
        ],
        False
    )

    kernel_group_start_pattern = r"ncclGroupStart\(\): pid (\d+)"
    kernel_group_end_pattern = r"ncclGroupEnd\(\): pid (\d+)"
    kernel_group_start_info = data[
        data["text"].str.match(kernel_group_start_pattern)
    ].copy()
    kernel_group_end_info = data[
        data["text"].str.match(kernel_group_end_pattern)
    ].copy()
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
        group_ends = group[~group["isStart"]].rename(columns={"end": "group_end"})[
            ["groupId", "group_end"]
        ]
        group = group_starts.merge(group_ends, on=["groupId"], how="inner")
        group = group[group["groupId"] != -1]
        group_start_end_grouped[gpu] = group

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
        p2p_kernels = p2p_kernel_grouped[gpu]
        if len(p2p_comm) > 0:
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
            # p2p_kernels = p2p_kernels.sort_values(by="start").reset_index(drop=True)
            # # check if the length of p2p_comm and p2p_kernels are the same
            # if len(p2p_comm) != len(p2p_kernels):
            #     raise ValueError(
            #         f"Mismatch in number of P2P comm events and P2P kernel events on GPU {gpu}: {len(p2p_comm)} vs {len(p2p_kernels)}"
            #     )
            # p2p_kernels["association"] = p2p_comm["eventId"].to_numpy()
            p2p_kernel_grouped[gpu] = p2p_kernels
            # comm_starts = p2p_comm["start"].to_numpy()
            # comm_ends = np.concat([comm_starts[1:], np.array([np.iinfo(np.int64).max])])
            # p2p_kernel_starts = p2p_kernels["start"].to_numpy()
            # p2p_kernels["association"] = _associate_events(comm_starts, comm_ends, p2p_comm["eventId"].to_numpy(), p2p_kernel_starts)
            # p2p_kernel_grouped[gpu] = p2p_kernels

    if len(coll_info_grouped) == 0:
        coll_info_grouped[("0", 0)] = pd.DataFrame(
            columns=list(coll_info_data.columns) + ["association"]
        )
    if len(coll_kernel_grouped) == 0:
        coll_kernel_grouped[("0", 0)] = pd.DataFrame(
            columns=list(coll_kernel_data.columns) + ["association"]
        )
    if len(p2p_kernel_grouped) == 0:
        p2p_kernel_grouped[("0", 0)] = pd.DataFrame(
            columns=list(p2p_kernel_data.columns) + ["association"]
        )
    
    comm_data = pd.concat((df.drop(columns=["commHash"], inplace=True) for df in comm_grouped.values()), ignore_index=True)
    coll_info_data = pd.concat((df.drop(columns=["eventId", "start", "end", "commHash", "stream", "nodeId", "pid"], inplace=True) for df in coll_info_grouped.values()), ignore_index=True)
    coll_kernel_data = pd.concat((df.drop(columns=["eventId", "start", "end", "nodeId", "pid"], inplace=True) for df in coll_kernel_grouped.values()), ignore_index=True)
    p2p_kernel_data = pd.concat((df.drop(columns=["eventId", "start", "end", "nodeId", "pid"], inplace=True) for df in p2p_kernel_grouped.values()), ignore_index=True)
    
    comm_grouped = {k: v.drop(columns=["commHash"], inplace=True) for k, v in comm_grouped.items()}
    coll_info_grouped = {k: v.drop(columns=["eventId", "start", "end", "commHash", "stream", "nodeId", "pid"], inplace=True) for k, v in coll_info_grouped.items()}
    coll_kernel_grouped = {k: v.drop(columns=["eventId", "start", "end", "nodeId", "pid"], inplace=True) for k, v in coll_kernel_grouped.items()}
    p2p_kernel_grouped = {k: v.drop(columns=["eventId", "start", "end", "nodeId", "pid"], inplace=True) for k, v in p2p_kernel_grouped.items()}
    # get the events grouped by stream
    if profiling_interval is not None:
        logger.info("filtering events by profiling intervals")
        comm_data = filter_time(profiling_interval, comm_data)
        coll_info_data = coll_info_data.merge(
            comm_data[["eventId"]],
            left_on=["association"],
            right_on=["eventId"],
            how="inner",
            suffixes=("", "_comm"),
        ).drop(columns=["eventId_comm"])

        coll_kernel_data = coll_kernel_data.merge(
            comm_data[["eventId"]],
            left_on=["association"],
            right_on=["eventId"],
            how="inner",
            suffixes=("", "_comm"),
        ).drop(columns=["eventId_comm"])

        p2p_kernel_data = p2p_kernel_data.merge(
            comm_data[["eventId"]],
            left_on=["association"],
            right_on=["eventId"],
            how="inner",
            suffixes=("", "_comm"),
        ).drop(columns=["eventId_comm"])
    return comm_data, coll_info_data, coll_kernel_data, p2p_kernel_data


def associate_kernel_to_nvtx(
    comm_data: pd.DataFrame,
    kernel_events: pd.DataFrame,
    profiling_interval: pd.DataFrame = None,
):
    if profiling_interval is not None:
        logger.info("filtering kernel events by profiling intervals")
        kernel_events = filter_time(profiling_interval, kernel_events)
    logger.info("associating kernels to nvtx events")
    kernel_df_grouped = {
        name: group for name, group in kernel_events.groupby(["nodeId", "pid"])
    }
    comm_grouped = {name: group for name, group in comm_data.groupby(["nodeId", "pid"])}
    kernels_list = []
    for gpu in tqdm(kernel_df_grouped.keys()):
        collective_labels = {
            "AllGather": "A",
            "AllReduce": "B",
            "Broadcast": "C",
            "ReduceScatter": "D",
            "Recv": "E",
            "Send": "E",
            "SendRecv": "E",
        }
        kernels = kernel_df_grouped[gpu].sort_values(by="start").reset_index(drop=True)
        nvtxs = comm_grouped[gpu].sort_values(by="start").reset_index(drop=True)
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
        kernels["label"] = kernels["collective"].map(collective_labels)
        nvtxs["label"] = nvtxs["collective"].map(collective_labels)

        kernel_stream_collectives = (
            kernels.groupby(["streamId"])
            .agg(
                {
                    "label": lambda x: "".join(x),
                    "start": "first",
                }
            )
            .reset_index()
        )
        kernel_stream_collectives["fingerPrint"] = kernel_stream_collectives[
            "label"
        ].map(lambda x: hex(hash(x) & 0xFFFFFFFFFFFFFFFF))
        # give index based on the start time to avoid hash collision
        kernel_stream_collectives = kernel_stream_collectives.sort_values(by=["fingerPrint", "start"]).reset_index(drop=True)
        kernel_stream_collectives["index"] = kernel_stream_collectives.groupby("fingerPrint").cumcount()
        

        nvtx_stream_collectives = (
            nvtxs.groupby(["stream"])
            .agg(
                {
                    "label": lambda x: "".join(x),
                    "start": "first",
                }
            )
            .reset_index()
        )
        nvtx_stream_collectives["fingerPrint"] = nvtx_stream_collectives["label"].map(
            lambda x: hex(hash(x) & 0xFFFFFFFFFFFFFFFF)
        )
        nvtx_stream_collectives = nvtx_stream_collectives.sort_values(by=["fingerPrint", "start"]).reset_index(drop=True)
        nvtx_stream_collectives["index"] = nvtx_stream_collectives.groupby("fingerPrint").cumcount()

        stream_correspondence = kernel_stream_collectives.merge(
            nvtx_stream_collectives,
            on=["fingerPrint", "index"],
            suffixes=("_kernel", "_nvtx"),
            how="outer",
        )
        if len(stream_correspondence) != max(
            len(kernel_stream_collectives), len(nvtx_stream_collectives)
        ):
            raise ValueError("Mismatch in number of unique stream fingerprints")
        # check for unmatched streams
        unmatched_kernel_streams = stream_correspondence[
            stream_correspondence["stream"].isna()
        ]
        if len(unmatched_kernel_streams) != 0:
            logger.error(
                f"GPU {gpu}: unmatched kernel streams: {unmatched_kernel_streams['streamId'].tolist()}"
            )
            for row in unmatched_kernel_streams.itertuples():
                logger.error(
                    f"  kernel streamId: {row.streamId}, label: {row.label_kernel}, fingerprint: {row.fingerPrint}"
                )
            raise ValueError("Unmatched kernel streams found")

        nvtxs["inStreamEventId"] = nvtxs.groupby("stream").cumcount()
        kernels["inStreamEventId"] = kernels.groupby("streamId").cumcount()
        kernels = (
            kernels.merge(
                stream_correspondence[["streamId", "stream"]],
                on=["streamId"],
                how="left",
            )
            .merge(
                nvtxs[["stream", "inStreamEventId", "eventId"]],
                on=["stream", "inStreamEventId"],
                how="left",
            )
            .drop(columns=["inStreamEventId", "stream", "label"])
            .rename(columns={"eventId": "association"})
        )

        kernel_df_grouped[gpu] = kernels
        nvtxs = (
            nvtxs.drop(columns=["label", "inStreamEventId", "start", "end"])
            .merge(
                kernels[["start", "end", "association"]],
                left_on=["eventId"],
                right_on=["association"],
            )
            .drop(columns=["association"])
        )

        dropped_nvtxs = dropped_nvtxs.drop(columns=["start", "end"]).merge(
            nvtxs[["end", "groupId"]], on=["groupId"], how="left"
        )

        dropped_nvtxs["start"] = dropped_nvtxs[
            "end"
        ]  # assign the end time of the previous nvtx as the start time of the dropped nvtx
        nvtxs = (
            pd.concat([nvtxs, dropped_nvtxs], ignore_index=True)
            .sort_values(by=["start"])
            .reset_index(drop=True)
        )

        comm_grouped[gpu] = nvtxs

    kernel_events = pd.concat(list(kernel_df_grouped.values()), ignore_index=True)
    comm_data = pd.concat(list(comm_grouped.values()), ignore_index=True)
    return comm_data, kernel_events
    
def add_context_parallelism(comm_data: pd.DataFrame):
    logger.info("adding context parallelism information")
    collective_labels = {
        "AllGather": "A",
        "AllReduce": "B",
        "Broadcast": "C",
        "ReduceScatter": "D",
        "Recv": "E",
        "Send": "F"
    }

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

    def get_rule(label_seq):
        if rule_dp(label_seq):
            return "DP"
        elif rule_pp(label_seq):
            return "PP"
        else:
            return "Other"

    comm_data = comm_data.copy().sort_values("start").reset_index(drop=True)
    comm_data["label"] = comm_data["collective"].map(collective_labels)
    comm_grouped = (
        comm_data.groupby(["nodeId", "pid", "stream"])
        .agg(label_seq=("label", lambda x: "".join(x)))
        .reset_index()
    )
    comm_grouped["parallelism"] = comm_grouped["label_seq"].map(get_rule)
    return comm_data.merge(
        comm_grouped[["nodeId", "pid", "stream", "parallelism"]],
        on=["nodeId", "pid", "stream"],
        how="left",
    ).drop(columns=["label"])



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
