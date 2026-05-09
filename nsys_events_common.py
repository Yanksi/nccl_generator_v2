"""
Common utilities for NSYS event processing.

This module contains shared code between nsys_events.py (sequential) and 
nsys_events_dask.py (parallel/Dask-based) implementations.
"""

import logging
import pathlib
import re
import sqlite3
from collections import defaultdict
from typing import Dict, List, Tuple

import numba
import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger("nsys_events")


def find_all_traces(directory):
    """Find all trace files that end with .sqlite in the given directory."""
    return list(pathlib.Path(directory).rglob("*.sqlite"))


def extract_node_id(trace_name: str) -> str:
    """Extract node ID from supported NSYS sqlite filename formats."""
    m = re.search(r"nid(\d+)", trace_name)
    if m is not None:
        return m.group(1)
    m = re.search(r"profile_(\d+)_(\d+)_(\d+)\.sqlite$", trace_name)
    if m is not None:
        return m.group(2)
    raise ValueError(f"Could not extract node ID from trace filename: {trace_name}")


def convert_numeric(
    data_frame: pd.DataFrame, signed_columns: List[str], unsigned_columns: List[str]
) -> pd.DataFrame:
    """Convert columns to signed/unsigned integer types."""
    if len(signed_columns) > 0:
        data_frame[signed_columns] = data_frame[signed_columns].astype("Int64")
    if len(unsigned_columns) > 0:
        data_frame[unsigned_columns] = data_frame[unsigned_columns].astype("UInt64")
    return data_frame


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

# Expected schemas for each category (columns after processing)
# Used for creating properly-typed empty DataFrames
CATEGORY_SCHEMAS = {
    "comm_info": {
        "start": "Int64", "end": "Int64", "nodeId": "object",
        "commHash": "object", "commId": "object", "rank": "UInt64", "nRanks": "UInt64", "pid": "UInt64"
    },
    "comm_ring": {
        "start": "Int64", "end": "Int64", "nodeId": "object",
        "commHash": "object", "channelId": "UInt64", "prevRank": "UInt64", "myRank": "UInt64", "nextRank": "UInt64", "pid": "UInt64"
    },
    "comm_tree": {
        "start": "Int64", "end": "Int64", "nodeId": "object",
        "commHash": "object", "channelId": "UInt64", "child1Rank": "Int64", "child2Rank": "Int64", "child3Rank": "Int64", "myRank": "UInt64", "parentRank": "Int64", "pid": "UInt64"
    },
    "profile_start": {"start": "Int64", "nodeId": "object", "pid": "Int64"},
    "profile_end": {"end": "Int64", "nodeId": "object", "pid": "Int64"},
    "group_start": {"start": "Int64", "end": "Int64", "nodeId": "object", "pid": "Int64", "isStart": "bool"},
    "group_end": {"start": "Int64", "end": "Int64", "nodeId": "object", "pid": "Int64", "isStart": "bool"},
    "coll_kernel": {
        "start": "Int64", "end": "Int64", "nodeId": "object",
        "count": "UInt64", "chunkCount": "UInt64", "workCount": "UInt64", "lastChunkCount": "UInt64",
        "workOffset": "UInt64", "sendbuff": "UInt64", "recvbuff": "UInt64", "pid": "UInt64"
    },
    "p2p_kernel": {
        "start": "Int64", "end": "Int64", "nodeId": "object",
        "Bytes": "UInt64", "p2pType": "object", "peer": "UInt64", "proto": "UInt64",
        "countHi32": "UInt64", "countLo32": "UInt64", "chunkSize": "UInt64", "pid": "UInt64"
    },
    "comm": {
        "start": "Int64", "end": "Int64", "nodeId": "object",
        "collective": "object", "commHash": "object", "stream": "object", "pid": "Int64"
    },
    "coll_info": {
        "start": "Int64", "end": "Int64", "nodeId": "object",
        "collType": "UInt64", "root": "UInt64", "redOp": "UInt64", "algo": "UInt64", "proto": "UInt64",
        "commHash": "object", "stream": "object", "data_size": "UInt64", "type_size": "UInt64",
        "chunkSteps": "UInt64", "sliceSteps": "UInt64", "stepSize": "UInt64", "pid": "UInt64"
    },
}

ALL_CATEGORIES = list(CATEGORY_SCHEMAS.keys())


def make_empty_df(category: str) -> pd.DataFrame:
    """Create an empty DataFrame with the correct schema for a category."""
    schema = CATEGORY_SCHEMAS[category]
    return pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in schema.items()})


def categorize_nvtx_text(text) -> str:
    """Categorize a single NVTX text string. Returns category name or 'other'."""
    if text is None or pd.isna(text):
        return "other"
    for category, pattern in NVTX_PATTERNS.items():
        if re.match(pattern, text):
            return category
    return "other"


def process_nvtx_by_category(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Process NVTX events by category, extracting fields and converting types.
    
    Handles empty DataFrames by ensuring proper column schema is maintained.
    """
    result = {}
    
    # comm_info
    comm_info = df[df["category"] == "comm_info"].copy()
    if len(comm_info) > 0:
        comm_info[["commHash", "commId", "rank", "nRanks", "pid"]] = comm_info["text"].str.extract(NVTX_PATTERNS["comm_info"])
        comm_info[["rank", "nRanks", "pid"]] = comm_info[["rank", "nRanks", "pid"]].astype("UInt64")
        comm_info = comm_info.drop(columns=["text", "category"])
        result["comm_info"] = comm_info
    else:
        result["comm_info"] = make_empty_df("comm_info")
    
    # comm_ring
    comm_ring = df[df["category"] == "comm_ring"].copy()
    if len(comm_ring) > 0:
        comm_ring[["commHash", "channelId", "prevRank", "myRank", "nextRank", "pid"]] = comm_ring["text"].str.extract(NVTX_PATTERNS["comm_ring"])
        comm_ring[["channelId", "prevRank", "myRank", "nextRank", "pid"]] = comm_ring[["channelId", "prevRank", "myRank", "nextRank", "pid"]].astype("UInt64")
        comm_ring = comm_ring.drop(columns=["text", "category"])
        result["comm_ring"] = comm_ring
    else:
        result["comm_ring"] = make_empty_df("comm_ring")
    
    # comm_tree
    comm_tree = df[df["category"] == "comm_tree"].copy()
    if len(comm_tree) > 0:
        comm_tree[["commHash", "channelId", "child1Rank", "child2Rank", "child3Rank", "myRank", "parentRank", "pid"]] = comm_tree["text"].str.extract(NVTX_PATTERNS["comm_tree"])
        comm_tree[["child1Rank", "child2Rank", "child3Rank", "parentRank"]] = comm_tree[["child1Rank", "child2Rank", "child3Rank", "parentRank"]].astype("Int64")
        comm_tree[["channelId", "myRank", "pid"]] = comm_tree[["channelId", "myRank", "pid"]].astype("UInt64")
        comm_tree = comm_tree.drop(columns=["text", "category"])
        result["comm_tree"] = comm_tree
    else:
        result["comm_tree"] = make_empty_df("comm_tree")
    
    # profile_start
    profile_start = df[df["category"] == "profile_start"].copy()
    if len(profile_start) > 0:
        profile_start["pid"] = profile_start["text"].str.extract(NVTX_PATTERNS["profile_start"])[0].astype("Int64")
        profile_start = profile_start.drop(columns=["text", "category", "end"])
        result["profile_start"] = profile_start
    else:
        result["profile_start"] = make_empty_df("profile_start")
    
    # profile_end
    profile_end = df[df["category"] == "profile_end"].copy()
    if len(profile_end) > 0:
        profile_end["pid"] = profile_end["text"].str.extract(NVTX_PATTERNS["profile_end"])[0].astype("Int64")
        profile_end = profile_end.drop(columns=["text", "category", "end"]).rename(columns={"start": "end"})
        result["profile_end"] = profile_end
    else:
        result["profile_end"] = make_empty_df("profile_end")
    
    # group_start
    group_start = df[df["category"] == "group_start"].copy()
    if len(group_start) > 0:
        group_start["pid"] = group_start["text"].str.extract(NVTX_PATTERNS["group_start"])[0].astype("Int64")
        group_start = group_start.drop(columns=["text", "category"])
        group_start["isStart"] = True
        result["group_start"] = group_start
    else:
        result["group_start"] = make_empty_df("group_start")
    
    # group_end
    group_end = df[df["category"] == "group_end"].copy()
    if len(group_end) > 0:
        group_end["pid"] = group_end["text"].str.extract(NVTX_PATTERNS["group_end"])[0].astype("Int64")
        group_end = group_end.drop(columns=["text", "category"])
        group_end["isStart"] = False
        result["group_end"] = group_end
    else:
        result["group_end"] = make_empty_df("group_end")
    
    # coll_kernel
    coll_kernel = df[df["category"] == "coll_kernel"].copy()
    if len(coll_kernel) > 0:
        coll_kernel[["count", "chunkCount", "workCount", "lastChunkCount", "workOffset", "sendbuff", "recvbuff", "pid"]] = coll_kernel["text"].str.extract(NVTX_PATTERNS["coll_kernel"])
        coll_kernel[["count", "chunkCount", "workCount", "lastChunkCount", "workOffset", "sendbuff", "recvbuff", "pid"]] = coll_kernel[["count", "chunkCount", "workCount", "lastChunkCount", "workOffset", "sendbuff", "recvbuff", "pid"]].astype("UInt64")
        coll_kernel = coll_kernel.drop(columns=["text", "category"])
        result["coll_kernel"] = coll_kernel
    else:
        result["coll_kernel"] = make_empty_df("coll_kernel")
    
    # p2p_kernel
    p2p_kernel = df[df["category"] == "p2p_kernel"].copy()
    if len(p2p_kernel) > 0:
        p2p_kernel[["Bytes", "p2pType", "peer", "proto", "countHi32", "countLo32", "chunkSize", "pid"]] = p2p_kernel["text"].str.extract(NVTX_PATTERNS["p2p_kernel"])
        p2p_kernel[["Bytes", "peer", "proto", "countHi32", "countLo32", "chunkSize", "pid"]] = p2p_kernel[["Bytes", "peer", "proto", "countHi32", "countLo32", "chunkSize", "pid"]].astype("UInt64")
        p2p_kernel = p2p_kernel.drop(columns=["text", "category"])
        result["p2p_kernel"] = p2p_kernel
    else:
        result["p2p_kernel"] = make_empty_df("p2p_kernel")
    
    # comm
    comm = df[df["category"] == "comm"].copy()
    if len(comm) > 0:
        comm[["collective", "commHash", "stream", "pid"]] = comm["text"].str.extract(NVTX_PATTERNS["comm"])
        comm["pid"] = comm["pid"].astype("Int64")
        comm = comm.drop(columns=["text", "category"])
        result["comm"] = comm
    else:
        result["comm"] = make_empty_df("comm")
    
    # coll_info
    coll_info = df[df["category"] == "coll_info"].copy()
    if len(coll_info) > 0:
        coll_info[["collType", "root", "redOp", "algo", "proto", "commHash", "stream", "data_size", "type_size", "chunkSteps", "sliceSteps", "stepSize", "pid"]] = coll_info["text"].str.extract(NVTX_PATTERNS["coll_info"])
        coll_info[["collType", "root", "redOp", "algo", "proto", "data_size", "type_size", "chunkSteps", "sliceSteps", "stepSize", "pid"]] = coll_info[["collType", "root", "redOp", "algo", "proto", "data_size", "type_size", "chunkSteps", "sliceSteps", "stepSize", "pid"]].astype("UInt64")
        coll_info = coll_info.drop(columns=["text", "category"])
        result["coll_info"] = coll_info
    else:
        result["coll_info"] = make_empty_df("coll_info")
    
    return result


def read_kernel_event_file(trace_file) -> pd.DataFrame:
    """Read NCCL kernel events from a single sqlite trace file."""
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
    df_tmp.drop(columns=["value"], inplace=True)
    return df_tmp


def read_nvtx_event_file(trace_file) -> Dict[str, pd.DataFrame]:
    """Read NVTX events from a single sqlite trace file and categorize them."""
    node_id = extract_node_id(trace_file.name)
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
        return {cat: make_empty_df(cat) for cat in ALL_CATEGORIES}
    df_tmp["nodeId"] = node_id
    df_tmp["category"] = df_tmp["text"].apply(categorize_nvtx_text)
    return process_nvtx_by_category(df_tmp)


# =============================================================================
# Numba-accelerated functions
# =============================================================================

def safe_numba(func):
    """Wrapper that falls back to pure Python if numba JIT fails."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return func.py_func(*args, **kwargs)
    return wrapper


@safe_numba
@numba.njit
def _associate_events(interval_starts, interval_ends, interval_id, events_time):
    """Associate events with intervals based on time overlap."""
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


@safe_numba
@numba.njit
def _associate_start_ends(sequence, internal_groups=True):
    """Associate start/end events into groups."""
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


# =============================================================================
# Data processing functions
# =============================================================================

def filter_time_single(profiling_interval: Tuple[int, int], df: pd.DataFrame) -> pd.DataFrame:
    """Filter events by a single profiling interval."""
    profile_start, profile_end = profiling_interval
    return df[(df["start"] < profile_end) & (df["end"] > profile_start)].reset_index(drop=True)

def filter_time(
    profiling_interval: Dict[Tuple[str, int], Tuple[int, int]],
    data: Dict[Tuple[str, int], pd.DataFrame]
) -> Dict[Tuple[str, int], pd.DataFrame]:
    """Filter events by profiling intervals."""
    result_dfs = {}
    logger.info("filtering events by profiling intervals")
    for gpu, gpu_df in tqdm(data.items(), total=len(data)):
        if gpu not in profiling_interval:
            logger.warning(f"GPU {gpu} has no profiling interval, keeping unfiltered events")
            result_dfs[gpu] = gpu_df.reset_index(drop=True)
            continue
        result_dfs[gpu] = filter_time_single(profiling_interval[gpu], gpu_df)
    return result_dfs


# =============================================================================
# Parallelism detection rules
# =============================================================================

COLLECTIVE_LABELS = {
    "AllGather": "A",
    "AllReduce": "B",
    "Broadcast": "C",
    "ReduceScatter": "D",
    "Recv": "E",
    "Send": "F",
    "SendRecv": "E",  # For kernel matching
}

# Labels used for kernel matching (Send/Recv/SendRecv all map to E)
COLLECTIVE_LABELS_KERNEL = {
    "AllGather": "A",
    "AllReduce": "B",
    "Broadcast": "C",
    "ReduceScatter": "D",
    "Recv": "E",
    "Send": "E",
    "SendRecv": "E",
}


def _rule_dp(label_seq: str) -> bool:
    """
    Check for fully sharded data parallelism pattern.
    The collective sequence should be like AAADDDAAADDD...
    """
    n_a = 0
    for c in label_seq:
        if c == "A":
            n_a += 1
        else:
            break
    if n_a == 0:
        return False
    for i in range(0, len(label_seq), 2 * n_a):
        if label_seq[i:i+n_a] != "A" * n_a:
            return False
        if label_seq[i+n_a:i+2*n_a] != "D" * n_a and i + n_a < len(label_seq):
            return False
    return True


def _rule_pp(label_seq: str) -> bool:
    """
    Check for pipeline parallelism pattern.
    Sequence of alternating E and F with even length.
    """
    if len(label_seq) % 2 != 0:
        return False
    first_two = label_seq[:2]
    if first_two not in ["EF", "FE"]:
        return False
    for i in range(0, len(label_seq), 2):
        if label_seq[i : i + 2] != first_two:
            return False
    return True


def _rule_pp2(label_seq: str) -> bool:
    """
    Check for pipeline parallelism variant.
    Sequence of alternating E and F with possible multiple Es or Fs in a row.
    e.g., EEEFFFEFEEFF
    """
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


PARALLELISM_RULES = [
    ("DP", _rule_dp),
    ("PP", _rule_pp),
    ("PP", _rule_pp2),
]


def get_parallelism_type(label_seq: str) -> str:
    """Determine parallelism type from collective label sequence."""
    for parallelism_name, rule_func in PARALLELISM_RULES:
        if rule_func(label_seq):
            return parallelism_name
    return "Other"


def add_context_parallelism(
    comm_datas: Dict[Tuple[str, int], pd.DataFrame]
) -> Dict[Tuple[str, int], pd.DataFrame]:
    """Add context parallelism information to communication data."""
    logger.info("adding context parallelism information")
    
    for gpu, comm_data in tqdm(comm_datas.items(), total=len(comm_datas)):
        comm_data = comm_data.sort_values("start").reset_index(drop=True)
        comm_data["label"] = comm_data["collective"].map(COLLECTIVE_LABELS)
        comm_grouped = (
            comm_data.groupby(["stream"])
            .agg(label_seq=("label", lambda x: "".join(x)))
            .reset_index()
        )
        comm_grouped["parallelism"] = comm_grouped["label_seq"].map(get_parallelism_type)
        comm_datas[gpu] = comm_data.merge(
            comm_grouped[["stream", "parallelism"]],
            on=["stream"],
            how="left",
        ).drop(columns=["label"])
    return comm_datas


# =============================================================================
# Kernel-to-NVTX association helpers
# =============================================================================

def _empty_kernel_times() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "eventId": pd.Series(dtype="Int64"),
            "start": pd.Series(dtype="Int64"),
            "end": pd.Series(dtype="Int64"),
        }
    )


def _split_grouped_nvtxs(nvtxs: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    non_grouped_nvtxs = nvtxs[nvtxs["groupId"] == -1]
    grouped_nvtxs = nvtxs[nvtxs["groupId"] != -1]

    first_nvtxs_in_group = grouped_nvtxs.groupby("groupId").first().reset_index()
    dropped_nvtxs = (
        grouped_nvtxs.groupby("groupId", group_keys=False)
        .apply(lambda x: x.iloc[1:])
        .reset_index(drop=True)
    )

    first_nvtxs = (
        pd.concat([non_grouped_nvtxs, first_nvtxs_in_group], ignore_index=True)
        .sort_values(by="start")
        .reset_index(drop=True)
    )
    return first_nvtxs, dropped_nvtxs


def _add_dropped_group_times(
    nvtxs: pd.DataFrame,
    dropped_nvtxs: pd.DataFrame,
    kernel_times: pd.DataFrame,
) -> pd.DataFrame:
    if len(dropped_nvtxs) == 0:
        return kernel_times[["eventId", "start", "end"]]

    first_event_times = nvtxs[nvtxs["groupId"] != -1][["eventId", "groupId"]].merge(
        kernel_times[["eventId", "end"]], on="eventId", how="left"
    )
    group_end_times = first_event_times.groupby("groupId")["end"].first().reset_index()
    dropped_times = dropped_nvtxs[["eventId", "groupId"]].merge(
        group_end_times, on="groupId", how="left"
    )
    dropped_times["start"] = dropped_times["end"]
    dropped_times = dropped_times[["eventId", "start", "end"]]

    return pd.concat([kernel_times, dropped_times], ignore_index=True)[
        ["eventId", "start", "end"]
    ]


def _process_one_gpu_kernels_by_stream(
    gpu,
    kernels: pd.DataFrame,
    nvtxs: pd.DataFrame,
) -> pd.DataFrame:
    """Use full NVTX stream annotations to match kernel streams to NVTX streams."""
    if len(kernels) == 0 or len(nvtxs) == 0:
        return _empty_kernel_times()

    nvtxs, dropped_nvtxs = _split_grouped_nvtxs(nvtxs)
    kernels = kernels.copy()
    nvtxs = nvtxs.copy()
    kernels["label"] = kernels["collective"].map(COLLECTIVE_LABELS_KERNEL)
    nvtxs["label"] = nvtxs["collective"].map(COLLECTIVE_LABELS_KERNEL)
    if kernels["label"].isna().any() or nvtxs["label"].isna().any():
        raise ValueError(f"Unknown collective label while matching GPU {gpu}")

    kernel_stream_collectives = (
        kernels.groupby(["streamId"])
        .agg({"label": lambda x: "".join(x), "start": "first"})
        .reset_index()
    )
    kernel_stream_collectives["fingerPrint"] = kernel_stream_collectives["label"].map(
        lambda x: hex(hash(x) & 0xFFFFFFFFFFFFFFFF)
    )
    kernel_stream_collectives = kernel_stream_collectives.sort_values(
        by=["fingerPrint", "start"]
    ).reset_index(drop=True)
    kernel_stream_collectives["index"] = kernel_stream_collectives.groupby(
        "fingerPrint"
    ).cumcount()

    nvtx_stream_collectives = (
        nvtxs.groupby(["stream"])
        .agg({"label": lambda x: "".join(x), "start": "first"})
        .reset_index()
    )
    nvtx_stream_collectives["fingerPrint"] = nvtx_stream_collectives["label"].map(
        lambda x: hex(hash(x) & 0xFFFFFFFFFFFFFFFF)
    )
    nvtx_stream_collectives = nvtx_stream_collectives.sort_values(
        by=["fingerPrint", "start"]
    ).reset_index(drop=True)
    nvtx_stream_collectives["index"] = nvtx_stream_collectives.groupby(
        "fingerPrint"
    ).cumcount()

    stream_correspondence = kernel_stream_collectives.merge(
        nvtx_stream_collectives,
        on=["fingerPrint", "index"],
        suffixes=("_kernel", "_nvtx"),
        how="outer",
    )
    if len(stream_correspondence) != max(
        len(kernel_stream_collectives), len(nvtx_stream_collectives)
    ):
        logger.debug(
            "nvtx_stream_collectives:\n%s\nkernel_stream_collectives:\n%s",
            nvtx_stream_collectives,
            kernel_stream_collectives,
        )
        raise ValueError(f"Mismatch in number of unique stream fingerprints {gpu}")

    unmatched_kernel_streams = stream_correspondence[
        stream_correspondence["stream"].isna()
    ]
    if len(unmatched_kernel_streams) != 0:
        for row in unmatched_kernel_streams.itertuples():
            logger.warning(
                "GPU %s: unmatched kernel streamId=%s label=%s fingerprint=%s",
                gpu,
                row.streamId,
                row.label_kernel,
                row.fingerPrint,
            )
        raise ValueError(f"Unmatched kernel streams found for GPU {gpu}")

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
    if kernels["association"].isna().any():
        missing = int(kernels["association"].isna().sum())
        raise ValueError(f"GPU {gpu}: {missing} kernels could not be stream-matched")

    kernel_times = kernels[["start", "end", "association"]].rename(
        columns={"association": "eventId"}
    )
    kernel_times["eventId"] = kernel_times["eventId"].astype("Int64")
    return _add_dropped_group_times(nvtxs, dropped_nvtxs, kernel_times)


def _process_one_gpu_kernels_by_time(
    gpu,
    kernels: pd.DataFrame,
    nvtxs: pd.DataFrame,
) -> pd.DataFrame:
    """Fallback for partial annotations: associate kernels to NVTX intervals by time."""
    if len(kernels) == 0 or len(nvtxs) == 0:
        return _empty_kernel_times()

    nvtxs, dropped_nvtxs = _split_grouped_nvtxs(nvtxs)
    nvtx_starts = nvtxs["start"].to_numpy(dtype=np.int64)
    nvtx_ends = nvtxs["end"].to_numpy(dtype=np.int64)
    nvtx_event_ids = nvtxs["eventId"].to_numpy(dtype=np.int64)
    kernel_starts = kernels["start"].to_numpy(dtype=np.int64)

    kernels = kernels.copy()
    kernels["association"] = _associate_events(
        nvtx_starts,
        nvtx_ends,
        nvtx_event_ids,
        kernel_starts,
    )

    unassociated_mask = kernels["association"] == -1
    if unassociated_mask.any():
        logger.warning(
            "GPU %s: %d kernels needed partial-annotation time fallback",
            gpu,
            int(unassociated_mask.sum()),
        )
        unassociated_times = kernels.loc[unassociated_mask, "start"].to_numpy(
            dtype=np.int64
        )
        associations = np.searchsorted(
            nvtx_starts, unassociated_times, side="right"
        ) - 1
        associations = np.clip(associations, 0, len(nvtx_event_ids) - 1)
        kernels.loc[unassociated_mask, "association"] = nvtx_event_ids[associations]

    kernel_times = (
        kernels.groupby("association")
        .agg(start=("start", "min"), end=("end", "max"))
        .reset_index()
        .rename(columns={"association": "eventId"})
    )
    kernel_times["eventId"] = kernel_times["eventId"].astype("Int64")
    return _add_dropped_group_times(nvtxs, dropped_nvtxs, kernel_times)


def process_one_gpu_kernels(
    gpu,
    kernels: pd.DataFrame,
    nvtxs: pd.DataFrame,
    profiling_interval: Tuple[int, int] = None,
) -> Tuple[Tuple[str, int], pd.DataFrame]:
    """
    Process kernel events for a single GPU and return eventId/start/end times.

    Full stream-fingerprint matching is used when all annotations are present.
    The time-based partial-annotation fallback is used only when stream matching
    proves that required stream annotations are missing.
    """
    if profiling_interval is not None:
        kernels = filter_time_single(profiling_interval, kernels)
        nvtxs = filter_time_single(profiling_interval, nvtxs)
    kernels = kernels.sort_values(by="start").reset_index(drop=True)
    nvtxs = nvtxs.sort_values(by="start").reset_index(drop=True)

    try:
        kernel_times = _process_one_gpu_kernels_by_stream(
            gpu, kernels.copy(), nvtxs.copy()
        )
    except ValueError as exc:
        logger.warning(
            "Full NVTX/kernel stream association failed for GPU %s; "
            "using partial-annotation fallback: %s",
            gpu,
            exc,
        )
        kernel_times = _process_one_gpu_kernels_by_time(gpu, kernels, nvtxs)

    return gpu, kernel_times[["eventId", "start", "end"]]
