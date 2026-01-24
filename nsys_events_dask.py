import logging
import os
import pathlib
import re
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numba
import dask
import dask.dataframe as dd
from dask import delayed
# dask.config.set(scheduler='processes')
from tqdm import tqdm
from tqdm.dask import TqdmCallback
from functools import partial

# Configure tqdm for batch job compatibility (explicit stderr, no buffering)
_tqdm_for_dask = partial(tqdm, file=sys.stderr, mininterval=0, dynamic_ncols=True)

# import modin.pandas as pd
import numpy as np
import pandas as pd

logger = logging.getLogger("nsys_events")
logging.basicConfig(level=logging.INFO)


def find_all_traces(directory):
    # find all trace files that ends with .sqlite
    return list(pathlib.Path(directory).rglob("*.sqlite"))

def read_kernel_event_file(trace_file):
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
    df_tmp.drop(columns=["value"], inplace=True)
    return df_tmp

def get_kernel_events(traces: List[os.PathLike]) -> dd.DataFrame:
    logger.info(f"querying for kernel events from {len(traces)} traces")
    dfs = []
    for trace_file in traces:
        dfs.append(delayed(read_kernel_event_file)(trace_file))
    
    meta = {
        "start": "Int64",
        "end": "Int64",
        "deviceId": "Int64",
        "streamId": "Int64",
        "pid": "Int64",
        "nodeId": "object",
        "collective": "string"
    }
    kernel_df = dd.from_delayed(dfs, meta=meta)
    return kernel_df

# All NVTX regex patterns - defined once globally
NVTX_PATTERNS = {
    "comm_info": r"commHash (0x[0-9a-f]+) commId (0x[0-9a-f]+) rank (\d+) nranks (\d+) pid (\d+)",
    "comm_ring": r"commHash (0x[0-9a-f]+) Rings \[(\d+)\] (\d+)->(\d+)->(\d+) pid (\d+)",
    "comm_tree": r"commHash (0x[0-9a-f]+) Trees \[(\d+)\] (-?\d+)/(-?\d+)/(-?\d+)->(-?\d+)->(-?\d+) pid (\d+)",
    "profile_start": r"nsys profiling start, pid: (\d+)",
    "profile_end": r"nsys profiling stopped, pid: (\d+)",
    "group_start": r"ncclGroupStart\(\): pid (\d+)",
    "group_end": r"ncclGroupEnd\(\): pid (\d+)",
    "coll_kernel": r"nWarps \d+ count (\d+) chunkCount (\d+) workCount (\d+) lastChunkCount (\d+) workOffset (\d+) sendbuff (\d+) recvbuff (\d+) pid (\d+)",
    "p2p_kernel": r"Bytes (\d+) nWarps \d+ p2pType (\d+) peer (\d+) proto (\d+) countHi32 (\d+) countLo32 (\d+) chunkSize (\d+) pid (\d+)",
    "comm": r"nccl([a-zA-Z]+)\(\): commHash (0x[0-9a-f]+), stream (0x[0-9a-f]+), data_size \d+, type_size \d+,.* pid (\d+)",
    "coll_info": r"collType (\d+) root (\d+) redOp (\d+) algo (\d+) proto (\d+) commHash (\S+) stream (\S+) data_size (\d+) type_size (\d+) chunkSize \d+ chunkCount \d+ chunkSteps (\d+) sliceSteps (\d+) stepSize (\d+) pid (\d+)",
}

def categorize_nvtx_text(text) -> str:
    """Categorize a single NVTX text string. Returns category name or 'other'."""
    if text is None or pd.isna(text):
        return "other"
    for category, pattern in NVTX_PATTERNS.items():
        if re.match(pattern, text):
            return category
    return "other"

def process_nvtx_by_category(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Process NVTX events by category, extracting fields and converting types."""
    result = {}
    
    # comm_info
    comm_info = df[df["category"] == "comm_info"].copy()
    if len(comm_info) > 0:
        comm_info[["commHash", "commId", "rank", "nRanks", "pid"]] = comm_info["text"].str.extract(NVTX_PATTERNS["comm_info"])
        comm_info[["rank", "nRanks", "pid"]] = comm_info[["rank", "nRanks", "pid"]].astype("UInt64")
        comm_info = comm_info.drop(columns=["text", "category"])
    result["comm_info"] = comm_info
    
    # comm_ring
    comm_ring = df[df["category"] == "comm_ring"].copy()
    if len(comm_ring) > 0:
        comm_ring[["commHash", "channelId", "prevRank", "myRank", "nextRank", "pid"]] = comm_ring["text"].str.extract(NVTX_PATTERNS["comm_ring"])
        comm_ring[["channelId", "prevRank", "myRank", "nextRank", "pid"]] = comm_ring[["channelId", "prevRank", "myRank", "nextRank", "pid"]].astype("UInt64")
        comm_ring = comm_ring.drop(columns=["text", "category"])
    result["comm_ring"] = comm_ring
    
    # comm_tree
    comm_tree = df[df["category"] == "comm_tree"].copy()
    if len(comm_tree) > 0:
        comm_tree[["commHash", "channelId", "child1Rank", "child2Rank", "child3Rank", "myRank", "parentRank", "pid"]] = comm_tree["text"].str.extract(NVTX_PATTERNS["comm_tree"])
        comm_tree[["child1Rank", "child2Rank", "child3Rank", "parentRank"]] = comm_tree[["child1Rank", "child2Rank", "child3Rank", "parentRank"]].astype("Int64")
        comm_tree[["channelId", "myRank", "pid"]] = comm_tree[["channelId", "myRank", "pid"]].astype("UInt64")
        comm_tree = comm_tree.drop(columns=["text", "category"])
    result["comm_tree"] = comm_tree
    
    # profile_start
    profile_start = df[df["category"] == "profile_start"].copy()
    if len(profile_start) > 0:
        profile_start["pid"] = profile_start["text"].str.extract(NVTX_PATTERNS["profile_start"])[0].astype("Int64")
        profile_start = profile_start.drop(columns=["text", "category", "end"])
    result["profile_start"] = profile_start
    
    # profile_end
    profile_end = df[df["category"] == "profile_end"].copy()
    if len(profile_end) > 0:
        profile_end["pid"] = profile_end["text"].str.extract(NVTX_PATTERNS["profile_end"])[0].astype("Int64")
        # Drop original 'end' column first to avoid duplicate column names after rename
        profile_end = profile_end.drop(columns=["text", "category", "end"]).rename(columns={"start": "end"})
    result["profile_end"] = profile_end
    
    # group_start
    group_start = df[df["category"] == "group_start"].copy()
    if len(group_start) > 0:
        group_start["pid"] = group_start["text"].str.extract(NVTX_PATTERNS["group_start"])[0].astype("Int64")
        group_start = group_start.drop(columns=["text", "category"])
        group_start["isStart"] = True
    result["group_start"] = group_start
    
    # group_end
    group_end = df[df["category"] == "group_end"].copy()
    if len(group_end) > 0:
        group_end["pid"] = group_end["text"].str.extract(NVTX_PATTERNS["group_end"])[0].astype("Int64")
        group_end = group_end.drop(columns=["text", "category"])
        group_end["isStart"] = False
    result["group_end"] = group_end
    
    # coll_kernel
    coll_kernel = df[df["category"] == "coll_kernel"].copy()
    if len(coll_kernel) > 0:
        coll_kernel[["count", "chunkCount", "workCount", "lastChunkCount", "workOffset", "sendbuff", "recvbuff", "pid"]] = coll_kernel["text"].str.extract(NVTX_PATTERNS["coll_kernel"])
        coll_kernel[["count", "chunkCount", "workCount", "lastChunkCount", "workOffset", "sendbuff", "recvbuff", "pid"]] = coll_kernel[["count", "chunkCount", "workCount", "lastChunkCount", "workOffset", "sendbuff", "recvbuff", "pid"]].astype("UInt64")
        coll_kernel = coll_kernel.drop(columns=["text", "category"])
    result["coll_kernel"] = coll_kernel
    
    # p2p_kernel
    p2p_kernel = df[df["category"] == "p2p_kernel"].copy()
    if len(p2p_kernel) > 0:
        p2p_kernel[["Bytes", "p2pType", "peer", "proto", "countHi32", "countLo32", "chunkSize", "pid"]] = p2p_kernel["text"].str.extract(NVTX_PATTERNS["p2p_kernel"])
        p2p_kernel[["Bytes", "peer", "proto", "countHi32", "countLo32", "chunkSize", "pid"]] = p2p_kernel[["Bytes", "peer", "proto", "countHi32", "countLo32", "chunkSize", "pid"]].astype("UInt64")
        p2p_kernel = p2p_kernel.drop(columns=["text", "category"])
    result["p2p_kernel"] = p2p_kernel
    
    # comm
    comm = df[df["category"] == "comm"].copy()
    if len(comm) > 0:
        comm[["collective", "commHash", "stream", "pid"]] = comm["text"].str.extract(NVTX_PATTERNS["comm"])
        comm["pid"] = comm["pid"].astype("Int64")
        comm = comm.drop(columns=["text", "category"])
    result["comm"] = comm
    
    # coll_info
    coll_info = df[df["category"] == "coll_info"].copy()
    if len(coll_info) > 0:
        coll_info[["collType", "root", "redOp", "algo", "proto", "commHash", "stream", "data_size", "type_size", "chunkSteps", "sliceSteps", "stepSize", "pid"]] = coll_info["text"].str.extract(NVTX_PATTERNS["coll_info"])
        coll_info[["collType", "root", "redOp", "algo", "proto", "data_size", "type_size", "chunkSteps", "sliceSteps", "stepSize", "pid"]] = coll_info[["collType", "root", "redOp", "algo", "proto", "data_size", "type_size", "chunkSteps", "sliceSteps", "stepSize", "pid"]].astype("UInt64")
        coll_info = coll_info.drop(columns=["text", "category"])
    result["coll_info"] = coll_info
    
    return result

def read_nvtx_event_file(trace_file):
    node_id = re.search(r"nid(\d+)", trace_file.name).group(1)
    try:
        conn = sqlite3.connect(trace_file)
        df_tmp = pd.read_sql_query(
            "SELECT start, end, text FROM NVTX_EVENTS",
            conn,
            dtype={"start": "Int64", "end": "Int64", "text": "string"},
        )
        conn.close()
    except Exception as e:
        logger.warning(f"Failed to read {trace_file}: {e}")
        return {cat: pd.DataFrame() for cat in ["comm_info", "comm_ring", "comm_tree", "profile_start", "profile_end", "group_start", "group_end", "coll_kernel", "p2p_kernel", "comm", "coll_info"]}
    df_tmp["nodeId"] = node_id
    # Categorize during read - this is parallelized across files
    df_tmp["category"] = df_tmp["text"].apply(categorize_nvtx_text)
    # Extract fields and convert types during read - also parallelized
    return process_nvtx_by_category(df_tmp)

def get_nvtx_events(traces: List[os.PathLike]) -> Dict[str, pd.DataFrame]:
    logger.info(f"querying for nvtx events from {len(traces)} traces")
    dfs = []
    for trace_file in traces:
        dfs.append(delayed(read_nvtx_event_file)(trace_file))
    
    logger.info("computing nvtx events in parallel")
    with TqdmCallback(desc="nvtx events", tqdm_class=_tqdm_for_dask):
        results = dask.compute(*dfs, scheduler="processes")
    
    # Combine results from all files
    combined = {}
    categories = ["comm_info", "comm_ring", "comm_tree", "profile_start", "profile_end", "group_start", "group_end", "coll_kernel", "p2p_kernel", "comm", "coll_info"]
    for cat in categories:
        cat_dfs = [r[cat] for r in results if len(r[cat]) > 0]
        if cat_dfs:
            combined[cat] = pd.concat(cat_dfs, ignore_index=True)
        else:
            combined[cat] = pd.DataFrame()
    
    return combined


def convert_numeric(data_frame: pd.DataFrame, signed_columns: List[str], unsigned_columns: List[str]) -> pd.DataFrame:
    if len(signed_columns) > 0:
        data_frame[signed_columns] = data_frame[signed_columns].astype("Int64")
    if len(unsigned_columns) > 0:
        data_frame[unsigned_columns] = data_frame[unsigned_columns].astype("UInt64")
    return data_frame


def get_communicator_info(data: Dict[str, pd.DataFrame]):
    logger.info("extracting communicator info")
    
    # Data is already extracted and typed
    comm_info = data["comm_info"].copy()
    comm_ring_info = data["comm_ring"].copy()
    comm_tree_info = data["comm_tree"].copy()
    
    comm_info = comm_info.drop(columns=["start", "end"]).drop_duplicates()
    comm_hash2id = comm_info[["nodeId", "commHash", "commId"]].drop_duplicates()

    comm_ring_info = (
        comm_ring_info.merge(comm_hash2id, on=["nodeId", "commHash"], how="left")
        .drop(columns=["commHash", "start", "end"])
        .drop_duplicates()
    )

    comm_tree_info = (
        comm_tree_info.merge(comm_hash2id, on=["nodeId", "commHash"], how="left")
        .drop(columns=["commHash", "start", "end"])
        .drop_duplicates()
    )
    return comm_info, comm_ring_info, comm_tree_info, data


def get_profiling_interval(data: Dict[str, pd.DataFrame]):
    logger.info("extracting profiling intervals")
    
    # Data is already extracted and typed
    profile_start_info = data["profile_start"].copy()
    profile_end_info = data["profile_end"].copy()
    
    result_df = profile_start_info.merge(profile_end_info, on=["nodeId", "pid"])[
        ["nodeId", "pid", "start", "end"]
    ]
    return {(row["nodeId"], row["pid"]): (row["start"], row["end"]) for _, row in result_df.iterrows()}, data


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


def get_event_info(data: Dict[str, pd.DataFrame], comm_info: pd.DataFrame = None):
    """
    Process pre-extracted NVTX data and associate events.
    
    Args:
        data: Dict of category -> DataFrame with pre-extracted fields from get_nvtx_events()
        comm_info: Optional communicator info DataFrame
    """
    logger.info("extracting event infos")
    
    # Get pre-processed DataFrames (already have fields extracted and types converted)
    kernel_group_start_info = data.get("group_start", pd.DataFrame()).copy()
    kernel_group_end_info = data.get("group_end", pd.DataFrame()).copy()
    coll_kernel_data = data.get("coll_kernel", pd.DataFrame()).copy()
    p2p_kernel_data = data.get("p2p_kernel", pd.DataFrame()).copy()
    comm_data = data.get("comm", pd.DataFrame()).copy()
    coll_info_data = data.get("coll_info", pd.DataFrame()).copy()
    
    # concat start and end info (isStart already set in process_nvtx_by_category)
    kernel_group_start_end = pd.concat(
        [kernel_group_start_info, kernel_group_end_info]
    )
    
    # Add comm_info mapping if provided
    if comm_info is not None and len(comm_data) > 0:
        comm_id_map = comm_info[["nodeId", "commHash", "commId"]].drop_duplicates()
        comm_data = comm_data.merge(
            comm_id_map, 
            on=["nodeId", "commHash"], 
            how="left"
        )

    kernel_group_start_end = (
        kernel_group_start_end
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
        # Add eventId here after sorting per GPU, or globally?
        # Original code added global eventId before anything.
        # But we need unique eventId for association.
        
        # If we add it per GPU, we must ensure it doesn't conflict? 
        # Actually association is PER GPU. So local eventId 0..N per GPU is fine?
        # Let's check: 
        # `coll_infos["association"] = _associate_events(..., coll_comm["eventId"].to_numpy(), coll_info_starts)`
        # `associate_kernel_to_nvtx` matches `nvtxs["eventId"]` with `kernels["association"]`.
        
        # Since logic is grouped by GPU, Per-GPU unique IDs are sufficient.
        comm["eventId"] = range(len(comm)) # 0..N

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
            p2p_kernel_grouped[gpu] = p2p_kernels
    
    comm_grouped = {k: v.drop(columns=["commHash", "nodeId", "pid"]) for k, v in comm_grouped.items()}
    coll_info_grouped = {k: v.drop(columns=["start", "end", "commHash", "stream", "nodeId", "pid"]) for k, v in coll_info_grouped.items()}
    coll_kernel_grouped = {k: v.drop(columns=["start", "end", "nodeId", "pid"]) for k, v in coll_kernel_grouped.items()}
    p2p_kernel_grouped = {k: v.drop(columns=["start", "end", "nodeId", "pid"]) for k, v in p2p_kernel_grouped.items()}
    return comm_grouped, coll_info_grouped, coll_kernel_grouped, p2p_kernel_grouped, data


def _process_one_gpu_kernels(gpu, kernels, nvtxs):
    """
    Process kernel events for a single GPU - matches kernels to NVTX events.
    Extracted as a module-level function for reuse and potential parallelization.
    """
    collective_labels = {
        "AllGather": "A",
        "AllReduce": "B",
        "Broadcast": "C",
        "ReduceScatter": "D",
        "Recv": "E",
        "Send": "E",
        "SendRecv": "E",
    }
    kernels = kernels.sort_values(by="start").reset_index(drop=True)
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
        raise ValueError(f"Mismatch in number of unique stream fingerprints {gpu}")
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
    return gpu, kernels, nvtxs


def associate_kernel_to_nvtx(
    comm_grouped: Dict[Tuple[str, int], pd.DataFrame],
    kernel_events: dd.DataFrame,
    profiling_interval: Dict = None,
):
    """
    Associate kernel events to NVTX events.
    
    Processes kernel events in a memory-efficient streaming manner
    by reading partition by partition instead of materializing all data at once.
    """
    import gc
    
    # Memory-efficient streaming approach: process partitions one at a time
    logger.info("computing kernel events (streaming by partition)")
    
    n_partitions = kernel_events.npartitions
    
    # Accumulator for kernel events per GPU
    kernel_events_by_gpu = defaultdict(list)
    
    # Process partitions in small batches to limit memory
    batch_size = min(4, n_partitions)  # Process 4 partitions at a time
    
    for i in tqdm(range(0, n_partitions, batch_size), desc="kernel event partitions"):
        # Get partitions for this batch
        partitions = []
        for j in range(i, min(i + batch_size, n_partitions)):
            partitions.append(kernel_events.get_partition(j))
        
        # Compute this batch (using threads to avoid serialization overhead)
        results = dask.compute(*partitions, scheduler="threads")
        
        # Group by GPU and accumulate
        for df in results:
            if len(df) == 0:
                continue
            for (nodeId, pid), gpu_df in df.groupby(["nodeId", "pid"]):
                # Normalize key types
                gpu_key = (str(nodeId), int(pid))
                kernel_events_by_gpu[gpu_key].append(gpu_df)
        
        # Free memory from this batch
        del results
        gc.collect()
    
    # Concatenate per-GPU and convert to dict format
    logger.info("concatenating kernel events per GPU")
    kernel_df_grouped = {}
    for gpu_key, dfs in kernel_events_by_gpu.items():
        if dfs:
            kernel_df_grouped[gpu_key] = pd.concat(dfs, ignore_index=True)
    
    # Free the accumulator
    del kernel_events_by_gpu
    gc.collect()
    
    # Apply profiling interval filter if provided
    if profiling_interval is not None:
        logger.info("filtering kernel events by profiling intervals")
        kernel_df_grouped = filter_time(profiling_interval, kernel_df_grouped)
    
    # Process each GPU
    logger.info("associating kernels to nvtx events")
    for gpu in tqdm(comm_grouped.keys()):
        if gpu not in kernel_df_grouped:
            continue
        
        gpu, kernels, nvtxs = _process_one_gpu_kernels(
            gpu, kernel_df_grouped[gpu], comm_grouped[gpu]
        )
        kernel_df_grouped[gpu] = kernels
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
    
    def process_one_context(gpu, comm_data):
        comm_data = comm_data.sort_values("start").reset_index(drop=True)
        comm_data["label"] = comm_data["collective"].map(collective_labels)
        comm_grouped = (
            comm_data.groupby(["stream"])
            .agg(label_seq=("label", lambda x: "".join(x)))
            .reset_index()
        )
        comm_grouped["parallelism"] = comm_grouped["label_seq"].map(get_rule)
        comm_data = comm_data.merge(
            comm_grouped[["stream", "parallelism"]],
            on=["stream"],
            how="left",
        ).drop(columns=["label"])
        return gpu, comm_data

    for gpu, comm_data in tqdm(comm_datas.items()):
        gpu, comm_data = process_one_context(gpu, comm_data)
        comm_datas[gpu] = comm_data
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
