"""
Parallel (Dask-based) implementation of NSYS event processing.

This module uses Dask for parallel processing of trace files and
multiprocessing for kernel association.
"""

import logging
import os
import re
import sys
from collections import defaultdict
from functools import partial
from typing import Dict, List, Tuple

import dask
from dask import delayed
from tqdm import tqdm
from tqdm.dask import TqdmCallback

import numpy as np
import pandas as pd

# Import shared utilities from common module
from nsys_events_common import (
    # Core utilities
    find_all_traces,
    convert_numeric,
    logger,
    # Patterns and schemas
    NVTX_PATTERNS,
    CATEGORY_SCHEMAS,
    ALL_CATEGORIES,
    make_empty_df,
    # NVTX processing
    categorize_nvtx_text,
    process_nvtx_by_category,
    read_kernel_event_file,
    read_nvtx_event_file,
    # Numba functions
    _associate_events,
    _associate_start_ends,
    # Data processing
    filter_time,
    add_context_parallelism,
    # Kernel association
    process_one_gpu_kernels,
    COLLECTIVE_LABELS,
    COLLECTIVE_LABELS_KERNEL,
)

# Configure tqdm for batch job compatibility (explicit stderr, no buffering)
_tqdm_for_dask = partial(tqdm, file=sys.stderr, mininterval=0, dynamic_ncols=True)


def get_kernel_events(traces: List[os.PathLike]):
    """Return traces list - actual reading is done in parallel during association."""
    return traces


def get_nvtx_events(traces: List[os.PathLike]) -> Dict[str, pd.DataFrame]:
    """Read and combine NVTX events from all trace files in parallel using Dask."""
    logger.info(f"querying for nvtx events from {len(traces)} traces")
    dfs = []
    for trace_file in traces:
        dfs.append(delayed(read_nvtx_event_file)(trace_file))
    
    logger.info("computing nvtx events in parallel")
    with TqdmCallback(desc="nvtx events", tqdm_class=_tqdm_for_dask):
        results = dask.compute(*dfs, scheduler="processes")
    
    # Combine results from all files
    combined = {}
    for cat in ALL_CATEGORIES:
        cat_dfs = [r[cat] for r in results]
        combined[cat] = pd.concat(cat_dfs, ignore_index=True)
    return combined


def get_communicator_info(data: Dict[str, pd.DataFrame]):
    """Extract communicator info from pre-processed NVTX data."""
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
    """Extract profiling intervals from pre-processed NVTX data."""
    logger.info("extracting profiling intervals")
    
    # Data is already extracted and typed
    profile_start_info = data["profile_start"].copy()
    profile_end_info = data["profile_end"].copy()
    
    result_df = profile_start_info.merge(profile_end_info, on=["nodeId", "pid"])[
        ["nodeId", "pid", "start", "end"]
    ]
    return {(row["nodeId"], row["pid"]): (row["start"], row["end"]) for _, row in result_df.iterrows()}, data


def get_event_info(data: Dict[str, pd.DataFrame], comm_info: pd.DataFrame = None):
    """
    Process pre-extracted NVTX data and associate events.
    
    Args:
        data: Dict of category -> DataFrame with pre-extracted fields from get_nvtx_events()
        comm_info: Optional communicator info DataFrame
    """
    logger.info("extracting event infos")
    
    # Get pre-processed DataFrames (already have fields extracted and types converted)
    kernel_group_start_info = data["group_start"].copy()
    kernel_group_end_info = data["group_end"].copy()
    coll_kernel_data = data["coll_kernel"].copy()
    p2p_kernel_data = data["p2p_kernel"].copy()
    comm_data = data["comm"].copy()
    coll_info_data = data["coll_info"].copy()
    
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
        # Per-GPU unique IDs are sufficient since association is per GPU
        comm["eventId"] = range(len(comm))

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


# Global variables for Pool initializer pattern in associate_kernel_to_nvtx
_KERNEL_ASSOC_COMM_DATA: Dict[Tuple[str, int], pd.DataFrame] = None


def _init_kernel_assoc_worker(comm_grouped):
    """Initialize worker with comm_grouped data via copy-on-write."""
    global _KERNEL_ASSOC_COMM_DATA
    _KERNEL_ASSOC_COMM_DATA = comm_grouped


def _process_trace_file(trace_file: os.PathLike) -> Dict[Tuple[str, int], pd.DataFrame]:
    """
    Read kernel events from ONE sqlite file and associate with NVTX events.
    
    Accesses comm_grouped via global _KERNEL_ASSOC_COMM_DATA (copy-on-write after fork).
    Returns lightweight Dict of (nodeId, pid) -> DataFrame with (eventId, start, end).
    """
    global _KERNEL_ASSOC_COMM_DATA
    
    # Read kernel events from this trace file
    kernel_df = read_kernel_event_file(trace_file)
    
    if len(kernel_df) == 0:
        return {}
    
    # Extract node ID from trace filename
    node_id = re.search(r"nid(\d+)", trace_file.name).group(1)
    
    results = {}
    
    # Process each GPU in this node
    for (nodeId, pid), gpu_kernels in kernel_df.groupby(["nodeId", "pid"]):
        gpu_key = (str(nodeId), int(pid))
        
        # Skip if no NVTX events for this GPU
        if gpu_key not in _KERNEL_ASSOC_COMM_DATA:
            continue
        
        if len(gpu_kernels) == 0:
            continue
        
        # Associate kernels with NVTX events - returns only (eventId, start, end)
        try:
            _, kernel_times = process_one_gpu_kernels(
                gpu_key, gpu_kernels, _KERNEL_ASSOC_COMM_DATA[gpu_key]
            )
            results[gpu_key] = kernel_times
        except Exception as e:
            logger.error(f"Failed to process GPU {gpu_key} in {trace_file}: {e}")
            raise
    
    return results


def associate_kernel_to_nvtx(
    comm_grouped: Dict[Tuple[str, int], pd.DataFrame],
    traces: List[os.PathLike],
    profiling_interval: Dict = None,  # Kept for API compatibility, but not used here
):
    """
    Associate kernel events to NVTX events using multiprocessing Pool.
    
    Uses Pool initializer pattern to share comm_grouped via copy-on-write after fork.
    Each worker reads kernel events from ONE sqlite file and returns lightweight
    (eventId, start, end) mappings. Main process merges these into comm_grouped.
    
    Args:
        comm_grouped: Dict of (nodeId, pid) -> NVTX DataFrame
        traces: List of paths to sqlite trace files
        profiling_interval: Unused - kept for API compatibility
    
    Returns:
        Updated comm_grouped with kernel start/end times
    """
    import multiprocessing as mp
    
    logger.info(f"associating kernel events to nvtx events from {len(traces)} traces (parallelized)")
    
    # Filter traces to only those with matching nodes
    node_ids = {gpu_key[0] for gpu_key in comm_grouped.keys()}
    filtered_traces = []
    for trace_file in traces:
        node_id = re.search(r"nid(\d+)", trace_file.name).group(1)
        if node_id in node_ids:
            filtered_traces.append(trace_file)
    
    if not filtered_traces:
        logger.warning("No traces to process - no matching nodes found")
        return comm_grouped
    
    logger.info(f"running {len(filtered_traces)} kernel association tasks in parallel")
    
    # Use Pool with initializer for copy-on-write sharing
    with mp.Pool(
        processes=min(len(filtered_traces), mp.cpu_count()),
        initializer=_init_kernel_assoc_worker,
        initargs=(comm_grouped,)
    ) as pool:
        results = list(tqdm(
            pool.imap_unordered(_process_trace_file, filtered_traces),
            total=len(filtered_traces),
            desc="kernel association"
        ))
    
    # Merge kernel times into comm_grouped
    # Each result is Dict[gpu_key, DataFrame with (eventId, start, end)]
    processed_gpus = set()
    for result in results:
        for gpu_key, kernel_times in result.items():
            # Get original NVTX data and merge with kernel times
            nvtx_df = comm_grouped[gpu_key]
            
            # Drop old start/end columns and merge new ones from kernel times
            nvtx_df = nvtx_df.drop(columns=["start", "end"], errors="ignore")
            nvtx_df = nvtx_df.merge(kernel_times, on="eventId", how="left")
            
            comm_grouped[gpu_key] = nvtx_df
            processed_gpus.add(gpu_key)
    
    # Log any GPUs that weren't processed (no kernel events found)
    unprocessed = set(comm_grouped.keys()) - processed_gpus
    if unprocessed:
        logger.warning(f"{len(unprocessed)} GPUs had no kernel events: {list(unprocessed)[:5]}...")

    return comm_grouped


if __name__ == "__main__":
    traces = find_all_traces("traces/Llama70B_N64_GPU256_TP1_PP8_DP32_70B_BS32/sqlite")
    nvtx_events = get_nvtx_events(traces)
    comm_info, comm_ring_info, comm_tree_info, nvtx_events = get_communicator_info(nvtx_events)
    profiling_interval, nvtx_events = get_profiling_interval(nvtx_events)
    comm_data, coll_info, coll_kernels, p2p_kernels, nvtx_events = get_event_info(
        nvtx_events, comm_info
    )
    # Pass traces directly - kernel events are read and processed in parallel tasks
    # Note: filter_time should be called after association (not during)
    comm_data = associate_kernel_to_nvtx(comm_data, traces)
    comm_data = filter_time(profiling_interval, comm_data)
    comm_data = add_context_parallelism(comm_data)
