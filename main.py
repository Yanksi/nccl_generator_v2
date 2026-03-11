# %%
import logging
import multiprocessing
import os
import re
import sys
from collections import defaultdict

# import aiofiles
import pathlib
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from nccl_comm import *
from nccl_primitives import *
from gpu import *
import argparse
import time

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# %%
def construct_communicators(
    comm_info: pd.DataFrame, comm_ring_info: pd.DataFrame, comm_tree_info: pd.DataFrame
) -> Tuple[Dict[str, Communicator], Dict[Tuple[str, int], GPUDevice]]:
    logger.info("constructing communicator objects")
    comm_info = comm_info.loc[:, ~comm_info.columns.duplicated()].copy()

    # construct GPUDevice objects
    gpus_df = comm_info[["nodeId", "pid"]].drop_duplicates()
    gpu_devices = {
        (row["nodeId"], row["pid"]): GPUDevice(rank=i, node_id=row["nodeId"], pid=row["pid"])
        for i, (_, row) in enumerate(gpus_df.iterrows())
    }
    # Store gpu_id (nodeId, pid) tuples for Communicator construction
    gpu_ids = [(row.nodeId, row.pid) for row in comm_info.itertuples(index=False)]
    comm_info["gpu_id"] = gpu_ids
    # Also keep reference to GPUDevice for later use
    comm_info["gpu"] = [gpu_devices[gid] for gid in gpu_ids]
    comm_gpus_df = (
        comm_info.sort_values("rank")
        .groupby(["commId"])
        .aggregate({"gpu_id": list})
        .reset_index()
    )
    communicators = {
        row["commId"]: Communicator(row["commId"], row["gpu_id"])
        for _, row in comm_gpus_df.iterrows()
    }

    comm_ring_info = comm_ring_info.sort_values("channelId")
    for _, row in comm_ring_info.iterrows():
        communicators[row["commId"]].add_ring_topo(
            row["myRank"], row["prevRank"], row["nextRank"]
        )

    comm_tree_info = comm_tree_info.sort_values("channelId")
    for _, row in comm_tree_info.iterrows():
        children = [
            c
            for c in [row["child1Rank"], row["child2Rank"], row["child3Rank"]]
            if c >= 0
        ]
        communicators[row["commId"]].add_tree_topo(
            row["myRank"], row["parentRank"], children
        )

    return communicators, gpu_devices


def _infer_channels_by_comm(
    comm_data: Dict[Tuple[str, int], pd.DataFrame],
    coll_kernels: Dict[Tuple[str, int], pd.DataFrame],
) -> Dict[str, int]:
    ch_by_comm: Dict[str, int] = defaultdict(lambda: 1)
    for gpu_id, gpu_coll_kernels in coll_kernels.items():
        gpu_comm = comm_data.get(gpu_id)
        if gpu_comm is None or len(gpu_comm) == 0 or len(gpu_coll_kernels) == 0:
            continue
        event_to_comm = (
            gpu_comm[["eventId", "commId"]]
            .dropna(subset=["eventId", "commId"])
            .drop_duplicates()
            .set_index("eventId")["commId"]
            .astype(str)
            .to_dict()
        )
        if len(event_to_comm) == 0:
            continue
        for event_id, ch_count in gpu_coll_kernels.groupby("association").size().items():
            comm_id = event_to_comm.get(event_id)
            if comm_id is None:
                continue
            ch_by_comm[comm_id] = max(ch_by_comm[comm_id], int(ch_count))
    return ch_by_comm


def synthesize_topology_from_comm_info(
    comm_info: pd.DataFrame,
    comm_data: Dict[Tuple[str, int], pd.DataFrame],
    coll_kernels: Dict[Tuple[str, int], pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if len(comm_info) == 0:
        return pd.DataFrame(), pd.DataFrame()

    ch_by_comm = _infer_channels_by_comm(comm_data, coll_kernels)
    ring_rows = []
    tree_rows = []

    for comm_id, group in comm_info.groupby("commId"):
        group = (
            group.dropna(subset=["rank", "nodeId", "pid"])
            .copy()
            .sort_values("rank")
            .drop_duplicates(subset=["rank"], keep="first")
        )
        if len(group) == 0:
            continue

        group["rank"] = group["rank"].astype(int)
        expected = list(range(len(group)))
        if group["rank"].tolist() != expected:
            logger.warning(
                "Non-dense communicator ranks for %s in comm_info; reindexing to [0..%d].",
                comm_id,
                len(group) - 1,
            )
            group = group.reset_index(drop=True)
            group["rank"] = np.arange(len(group), dtype=np.int64)

        nranks = len(group)
        nchannels = max(1, int(ch_by_comm.get(str(comm_id), 1)))
        for row in group.itertuples(index=False):
            rank = int(row.rank)
            for ch in range(nchannels):
                ring_rows.append(
                    {
                        "nodeId": row.nodeId,
                        "commId": comm_id,
                        "channelId": ch,
                        "prevRank": (rank - 1) % nranks,
                        "myRank": rank,
                        "nextRank": (rank + 1) % nranks,
                        "pid": int(row.pid),
                    }
                )
                child1 = 2 * rank + 1
                child2 = 2 * rank + 2
                tree_rows.append(
                    {
                        "nodeId": row.nodeId,
                        "commId": comm_id,
                        "channelId": ch,
                        "child1Rank": child1 if child1 < nranks else -1,
                        "child2Rank": child2 if child2 < nranks else -1,
                        "child3Rank": -1,
                        "myRank": rank,
                        "parentRank": (rank - 1) // 2 if rank > 0 else -1,
                        "pid": int(row.pid),
                    }
                )

    return pd.DataFrame(ring_rows), pd.DataFrame(tree_rows)


def parse_nccl_log_communicators(
    log_path: pathlib.Path,
    gpu_ids: List[Tuple[str, int]],
    allowed_comm_ids: set = None,
) -> pd.DataFrame:
    """
    Parse communicator init lines from NCCL runtime logs.
    """
    if log_path is None or not log_path.exists():
        return pd.DataFrame()

    pid_to_node = {int(pid): node_id for node_id, pid in gpu_ids}
    allowed_norm = None
    canonical_comm_id = {}
    if allowed_comm_ids:
        allowed_norm = set()
        for comm_id in allowed_comm_ids:
            cid = str(comm_id)
            allowed_norm.add(cid.lower())
            canonical_comm_id[cid.lower()] = cid

    init_re = re.compile(
        r"^[^:\s]+:(?P<pid>\d+):\d+\s+\[\d+\]\s+NCCL INFO\s+ncclCommInitRankConfig\s+"
        r"comm\s+\S+\s+rank\s+(?P<rank>\d+)\s+nranks\s+(?P<nranks>\d+).*?"
        r"commId\s+(?P<commid>0x[0-9a-fA-F]+)"
    )
    rows = []
    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            m = init_re.search(line)
            if m is None:
                continue
            pid = int(m.group("pid"))
            node_id = pid_to_node.get(pid)
            if node_id is None:
                continue
            comm_id_raw = m.group("commid")
            comm_norm = comm_id_raw.lower()
            if allowed_norm is not None and comm_norm not in allowed_norm:
                continue
            comm_id = canonical_comm_id.get(comm_norm, comm_id_raw)
            rows.append(
                {
                    "nodeId": node_id,
                    "commHash": comm_id,
                    "commId": comm_id,
                    "rank": int(m.group("rank")),
                    "nRanks": int(m.group("nranks")),
                    "pid": pid,
                }
            )

    if len(rows) == 0:
        return pd.DataFrame()

    parsed = pd.DataFrame(rows).drop_duplicates(subset=["commId", "rank", "pid"], keep="first")
    return parsed


def compare_comm_membership(parsed_comm_info: pd.DataFrame, inferred_comm_info: pd.DataFrame) -> None:
    if len(parsed_comm_info) == 0 or len(inferred_comm_info) == 0:
        return

    parsed_map = {
        comm_id: set(group["pid"].astype(int).tolist())
        for comm_id, group in parsed_comm_info.groupby("commId")
    }
    inferred_map = {
        comm_id: set(group["pid"].astype(int).tolist())
        for comm_id, group in inferred_comm_info.groupby("commId")
    }
    parsed_ids = set(parsed_map.keys())
    inferred_ids = set(inferred_map.keys())
    common = parsed_ids & inferred_ids
    matched = sum(1 for cid in common if parsed_map[cid] == inferred_map[cid])
    mismatched = len(common) - matched
    logger.info(
        "NCCL-log communicator comparison: common=%d matched=%d mismatched=%d missing_in_log=%d extra_in_log=%d",
        len(common),
        matched,
        mismatched,
        len(inferred_ids - parsed_ids),
        len(parsed_ids - inferred_ids),
    )


def synthesize_communicators_from_events(
    comm_data: Dict[Tuple[str, int], pd.DataFrame],
    coll_kernels: Dict[Tuple[str, int], pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build communicator/ring/tree tables when comm init NVTX markers are absent.
    Uses observed commId membership from communication events.
    """
    members_by_comm: Dict[str, set] = defaultdict(set)
    for gpu_id, gpu_comm in comm_data.items():
        if "commId" not in gpu_comm.columns or len(gpu_comm) == 0:
            continue
        for comm_id in gpu_comm["commId"].dropna().astype(str).unique():
            members_by_comm[comm_id].add(gpu_id)

    if len(members_by_comm) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    ch_by_comm = _infer_channels_by_comm(comm_data, coll_kernels)

    def _gpu_sort_key(gpu_id: Tuple[str, int]):
        node_id, pid = gpu_id
        node_sort = int(node_id) if str(node_id).isdigit() else str(node_id)
        return (node_sort, int(pid))

    comm_info_rows = []
    ring_rows = []
    tree_rows = []
    for comm_id, members in members_by_comm.items():
        members_sorted = sorted(members, key=_gpu_sort_key)
        nranks = len(members_sorted)
        nchannels = max(1, int(ch_by_comm.get(comm_id, 1)))

        for rank, (node_id, pid) in enumerate(members_sorted):
            comm_info_rows.append(
                {
                    "nodeId": node_id,
                    "commHash": comm_id,
                    "commId": comm_id,
                    "rank": rank,
                    "nRanks": nranks,
                    "pid": int(pid),
                }
            )
            for ch in range(nchannels):
                ring_rows.append(
                    {
                        "nodeId": node_id,
                        "commId": comm_id,
                        "channelId": ch,
                        "prevRank": (rank - 1) % nranks,
                        "myRank": rank,
                        "nextRank": (rank + 1) % nranks,
                        "pid": int(pid),
                    }
                )
                child1 = 2 * rank + 1
                child2 = 2 * rank + 2
                tree_rows.append(
                    {
                        "nodeId": node_id,
                        "commId": comm_id,
                        "channelId": ch,
                        "child1Rank": child1 if child1 < nranks else -1,
                        "child2Rank": child2 if child2 < nranks else -1,
                        "child3Rank": -1,
                        "myRank": rank,
                        "parentRank": (rank - 1) // 2 if rank > 0 else -1,
                        "pid": int(pid),
                    }
                )

    comm_info = pd.DataFrame(comm_info_rows)
    comm_ring_info = pd.DataFrame(ring_rows)
    comm_tree_info = pd.DataFrame(tree_rows)
    return comm_info, comm_ring_info, comm_tree_info


algo_mapping = {
    0: CollAlgo.TREE,
    1: CollAlgo.RING,
}
proto_mapping = {
    0: NCCLProto.LL,
    1: NCCLProto.LL128,
    2: NCCLProto.SIMPLE,
}

context_labels = {"Other": 0, "PP": 1, "DP": 2}


def parse_buffer_size(size_str: str) -> int:
    """
    Parse a buffer size string into bytes.
    Supports formats like: '1MB', '512KB', '8192', '1M', '512K'
    """
    if size_str is None:
        return None
    
    size_str = size_str.strip().upper()
    
    # Check for unit suffixes
    if size_str.endswith('MB'):
        return int(float(size_str[:-2]) * 1024 * 1024)
    elif size_str.endswith('KB'):
        return int(float(size_str[:-2]) * 1024)
    elif size_str.endswith('GB'):
        return int(float(size_str[:-2]) * 1024 * 1024 * 1024)
    elif size_str.endswith('M'):
        return int(float(size_str[:-1]) * 1024 * 1024)
    elif size_str.endswith('K'):
        return int(float(size_str[:-1]) * 1024)
    elif size_str.endswith('B'):
        return int(size_str[:-1])
    else:
        # Assume plain number in bytes
        return int(size_str)


# Global variables for worker processes (set via Pool initializer for copy-on-write efficiency)
_WORKER_GPU_DATA = None
_WORKER_GPU_ID2GOAL_RANK = None
_WORKER_OUTPUT_DIR = None
_WORKER_MERGED_STREAMS = None
_WORKER_WRITE_BUFFER_SIZE = None
_WORKER_MERGE_SUCCESS = None  # Shared array for tracking merge success per GPU
_WORKER_MERGE_BARRIER = None  # Barrier for synchronizing merge phase


def _init_worker(gpu_data_dict, gpu_id2goal_rank, output_dir, merged_streams, write_buffer_size,
                 merge_success_array=None, merge_barrier=None):
    """
    Initialize worker process with shared data.
    This is called once per worker at fork time, enabling copy-on-write memory sharing.
    """
    global _WORKER_GPU_DATA, _WORKER_GPU_ID2GOAL_RANK, _WORKER_OUTPUT_DIR, _WORKER_MERGED_STREAMS
    global _WORKER_WRITE_BUFFER_SIZE, _WORKER_MERGE_SUCCESS, _WORKER_MERGE_BARRIER
    _WORKER_GPU_DATA = gpu_data_dict
    _WORKER_GPU_ID2GOAL_RANK = gpu_id2goal_rank
    _WORKER_OUTPUT_DIR = output_dir
    _WORKER_MERGED_STREAMS = merged_streams
    _WORKER_WRITE_BUFFER_SIZE = write_buffer_size
    _WORKER_MERGE_SUCCESS = merge_success_array
    _WORKER_MERGE_BARRIER = merge_barrier


def process_gpu_chunk(args_tuple):
    """
    Worker function for parallel GPU initialization and goal generation.
    Processes a chunk of GPUs (not just one) for better load balancing.
    Uses global data set via initializer (copy-on-write friendly).
    
    Two-phase stream merging:
    - Phase 1: Try merge_streams() for all GPUs in chunk, record success/failure
    - Barrier: Wait for all workers to complete phase 1
    - Phase 2: If ALL GPUs across all workers succeeded, apply merged streams; otherwise skip
    
    Args:
        args_tuple: (gpu_chunk, nic, init_counter, gen_counter, counter_lock)
        gpu_chunk: List of (gpu_id, rank) tuples for this worker to process
    """
    gpu_chunk, nic, init_counter, gen_counter, counter_lock = args_tuple
    
    # Access data from globals (copy-on-write, no pickle overhead)
    gpu_data_dict = _WORKER_GPU_DATA
    gpu_id2goal_rank = _WORKER_GPU_ID2GOAL_RANK
    output_dir = _WORKER_OUTPUT_DIR
    merged_streams = _WORKER_MERGED_STREAMS
    merge_success_array = _WORKER_MERGE_SUCCESS
    merge_barrier = _WORKER_MERGE_BARRIER
    
    # Phase 1: Initialize all GPUs in chunk and attempt stream merging
    gpus = []  # List of (gpu, rank, merged_stream_or_none)
    for gpu_id, rank in gpu_chunk:
        gpu_data = gpu_data_dict[gpu_id]
        
        # Reconstruct GPUDevice in this process
        gpu = GPUDevice(rank=rank, node_id=gpu_data["node_id"], pid=gpu_data["pid"])
        
        # Initialize GPU from dataframes
        gpu.init_from_dfs(
            gpu_data["coll_info"],
            gpu_data["coll_kernels"],
            gpu_data["p2p_kernels"],
            gpu_data["comm_data"]
        )
        
        # Try to merge streams if requested
        merged_stream = None
        if merged_streams:
            merged_stream = gpu.merge_streams()
            # Record success/failure in shared array (1 = success, 0 = failure)
            merge_success_array[rank] = 1 if merged_stream is not None else 0
        
        gpus.append((gpu, rank, merged_stream))
        
        # Update init counter
        with counter_lock:
            init_counter.value += 1
    
    # Barrier: Wait for all workers to complete Phase 1 (merge attempts)
    if merged_streams and merge_barrier is not None:
        merge_barrier.wait()
    
    # Phase 2: Check if ALL GPUs succeeded in merging
    all_merge_success = True
    if merged_streams and merge_success_array is not None:
        # Check all entries in the shared array
        all_merge_success = all(merge_success_array[i] == 1 for i in range(len(merge_success_array)))
    
    # Phase 3: Generate goal files
    results = []
    for gpu, rank, merged_stream in gpus:
        # Apply merged streams only if ALL GPUs succeeded
        if merged_streams and all_merge_success and merged_stream is not None:
            gpu.streams = {"merged_stream": merged_stream}
        # else: keep original streams
        
        # Generate and write goal file
        output_file = output_dir / f"rank_{rank}.goal"
        buffering = _WORKER_WRITE_BUFFER_SIZE if _WORKER_WRITE_BUFFER_SIZE and _WORKER_WRITE_BUFFER_SIZE > 0 else -1
        with open(output_file, "w", buffering=buffering) as f:
            f.write(f"rank {rank} {{\n")
            for line in gpu.generate_goal_lines(gpu_id2goal_rank, nic=nic):
                f.write(f"{line}\n")
            f.write("}\n")
        
        # Update generation counter
        with counter_lock:
            gen_counter.value += 1
        
        results.append(rank)
    
    return results


def write_goal_for_gpu(args_tuple):
    """
    Worker function for parallel goal file writing.
    Writes goal for a single GPU to a separate file.
    
    Args:
        args_tuple: (gpu, rank, gpu_id2goal_rank, output_dir, merged_streams, nic, write_buffer_size)
    """
    gpu, rank, gpu_id2goal_rank, output_dir, merged_streams, nic, write_buffer_size = args_tuple
    
    if merged_streams:
        gpu.streams = {"stream_merged": gpu.merge_streams()}

    output_file = output_dir / f"rank_{rank}.goal"
    buffering = write_buffer_size if write_buffer_size and write_buffer_size > 0 else -1
    with open(output_file, "w", buffering=buffering) as f:
        f.write(f"rank {rank} {{\n")
        for line in gpu.generate_goal_lines(gpu_id2goal_rank, nic=nic):
            f.write(f"{line}\n")
        f.write("}\n")
    
    return rank


def concatenate_goal_files(output_dir: pathlib.Path, num_ranks: int, delete_parts: bool = True, 
                           write_buffer_size: int = None):
    """
    Concatenate individual rank goal files into a single output.goal file.
    
    Args:
        output_dir: Directory containing the rank_*.goal files
        num_ranks: Total number of ranks
        delete_parts: Whether to delete individual rank files after concatenation
        write_buffer_size: Buffer size for file writes (None = system default)
    """
    output_file = output_dir / "output.goal"
    buffering = write_buffer_size if write_buffer_size and write_buffer_size > 0 else -1
    with open(output_file, "w", buffering=buffering) as out_f:
        out_f.write(f"num_ranks {num_ranks}\n")
        for rank in range(num_ranks):
            rank_file = output_dir / f"rank_{rank}.goal"
            with open(rank_file, "r") as rank_f:
                out_f.write(rank_f.read())
            if delete_parts:
                rank_file.unlink()
    return output_file


if __name__ == "__main__":
    # %%
    # get the path of the current script
    script_start_time = time.time()
    parser = argparse.ArgumentParser(description="Process some paths.")
    parser.add_argument("--trace_dir", "-i", type=str, required=True, help="Directory containing trace files")
    parser.add_argument("--output_dir", "-o", type=str, required=True, help="Directory to save output files")
    parser.add_argument("--npkit_data_simple", "-s", type=str, required=True, help="Path to npkit data summary for Simple protocol")
    parser.add_argument("--npkit_data_ll", "-l", type=str, required=True, help="Path to npkit data summary for LL protocol")
    parser.add_argument("--merged", "-m", action='store_true', help="Whether the streams are merged")
    parser.add_argument("--parallel_generation", "-p", action='store_true', help="Whether to generate goal files in parallel")
    parser.add_argument("--concatenate", "-c", action='store_true', help="Whether to concatenate all traces into one")
    parser.add_argument("--delete_parts", "-r", action='store_true', help="Whether to delete part files after concatenation (will automatically enable concatenation)")
    parser.add_argument("--write_buffer_size", type=str, default=None, 
                        help="Write buffer size for goal file output (e.g., '1MB', '512KB', '8192'). "
                             "Larger buffers reduce I/O operations, useful for network storage.")
    parser.add_argument("--n_workers", "-w", type=int, default=os.cpu_count(), help="Maximum number of worker processes")
    parser.add_argument(
        "--nccl_log",
        type=str,
        default=None,
        help="Optional NCCL runtime log file to recover communicator init metadata.",
    )

    args = parser.parse_args()
    init_generation_flags(
        zero_price_reduction_copy_flag=False,
        zero_price_communication_flag=False,
        enable_intra_node_transfer_flag=False,
    )
    trace_dir = pathlib.Path(args.trace_dir).resolve()
    output_dir = pathlib.Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_streams = args.merged
    parallel_generation = args.parallel_generation
    n_workers = args.n_workers
    concatenate = args.concatenate
    delete_parts = args.delete_parts
    if delete_parts:
        concatenate = True  # must concatenate if deleting parts
    write_buffer_size = parse_buffer_size(args.write_buffer_size)
    if write_buffer_size:
        logger.info(f"Using write buffer size: {write_buffer_size} bytes ({write_buffer_size / 1024:.1f} KB)")

    npkit_data_simple = args.npkit_data_simple
    npkit_data_ll = args.npkit_data_ll
    nccl_log_path = pathlib.Path(args.nccl_log).resolve() if args.nccl_log else None
    if nccl_log_path is None:
        auto_logs = sorted(trace_dir.parent.glob("log-*.out"))
        if auto_logs:
            nccl_log_path = auto_logs[0].resolve()
    if parallel_generation:
        import dask
        dask.config.set(scheduler="processes", num_workers=n_workers)
        from nsys_events_dask import *
        from tqdm.dask import TqdmCallback
        from functools import partial
        # Configure tqdm for batch job compatibility
        _tqdm_for_dask = partial(tqdm, file=sys.stderr, mininterval=0, dynamic_ncols=True)
    else:
        from nsys_events import *

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    traces = find_all_traces(trace_dir)
    nvtx_events = get_nvtx_events(traces)
    
    # nvtx_events is now Dict[str, pd.DataFrame] with pre-processed data
    # (parallel I/O, regex categorization, field extraction, and type conversion already done)

    comm_info, comm_ring_info, comm_tree_info, nvtx_events = get_communicator_info(nvtx_events)
    
    comm_data, coll_info, coll_kernels, p2p_kernels, nvtx_events = get_event_info(nvtx_events, comm_info)
    if len(comm_info) == 0:
        logger.warning("No communicator init markers found; synthesizing communicators from comm events.")
        inferred_comm_info, inferred_ring_info, inferred_tree_info = synthesize_communicators_from_events(
            comm_data, coll_kernels
        )
        comm_info, comm_ring_info, comm_tree_info = (
            inferred_comm_info,
            inferred_ring_info,
            inferred_tree_info,
        )
        if nccl_log_path is not None and nccl_log_path.exists():
            allowed_comm_ids = set(inferred_comm_info["commId"].dropna().astype(str).unique())
            parsed_comm_info = parse_nccl_log_communicators(nccl_log_path, list(comm_data.keys()), allowed_comm_ids)
            if len(parsed_comm_info) > 0:
                compare_comm_membership(parsed_comm_info, inferred_comm_info)
                parsed_ids = set(parsed_comm_info["commId"].dropna().astype(str).unique())
                missing_ids = allowed_comm_ids - parsed_ids
                if missing_ids:
                    logger.warning(
                        "NCCL-log communicator parse missing %d inferred communicators; falling back for missing IDs.",
                        len(missing_ids),
                    )
                    comm_info = pd.concat(
                        [
                            parsed_comm_info,
                            inferred_comm_info[
                                inferred_comm_info["commId"].astype(str).isin(missing_ids)
                            ],
                        ],
                        ignore_index=True,
                    )
                else:
                    comm_info = parsed_comm_info
                comm_ring_info, comm_tree_info = synthesize_topology_from_comm_info(
                    comm_info, comm_data, coll_kernels
                )
                logger.info("Using NCCL log communicator metadata from %s", nccl_log_path)
            else:
                logger.warning("NCCL log parsing produced no communicator metadata: %s", nccl_log_path)

    profiling_interval, nvtx_events = get_profiling_interval(nvtx_events)
    
    del nvtx_events
    
    kernel_events = get_kernel_events(traces)
    try:
        comm_data = associate_kernel_to_nvtx(comm_data, kernel_events, profiling_interval)
    except Exception as e:
        logger.info("Kernel association failed with pre-filtering, retrying with post filtering")
        comm_data = associate_kernel_to_nvtx(comm_data, kernel_events)
        comm_data = filter_time(profiling_interval, comm_data)
    
    comm_data = add_context_parallelism(comm_data)

    communicators, gpu_devices = construct_communicators(
        comm_info, comm_ring_info, comm_tree_info
    )
    communicator_ids_numeric = [[i, comm_id] for i, comm_id in enumerate(comm_info["commId"].unique())]
    communicator_ids_numeric_df = pd.DataFrame(communicator_ids_numeric, columns=["comm_num_id", "commId"])

    comm_ops = {
        "AllReduce": AllReduce,
        "AllGather": AllGather,
        "ReduceScatter": ReduceScatter,
        "Broadcast": Broadcast,
        "Reduce": Reduce,
        "Send": Send,
        "Recv": Recv,
    }

    comm_op_ids = {name: i for i, name in enumerate(comm_ops.keys())}
    comm_op_ids["Send"] = comm_op_ids["Recv"]
    
    # Build gpu_id2goal_rank mapping
    gpu_id2goal_rank = {gpu.gpu_id: i for i, gpu in enumerate(gpu_devices.values())}
    
    # Initialize npkit data
    init_data(
        npkit_data_simple,
        npkit_data_ll,
    )

    _MASK = (1 << sys.hash_info.width) - 1
    def hash_sequnces(*sequences):
        return [hash(elems) & _MASK for elems in zip(*sequences)]

    def prepare_gpu_data(gpu_id, gpu):
        """Prepare dataframes for a single GPU, applying all transformations."""
        coll_info_gpu = coll_info[gpu_id].copy()
        coll_info_gpu["algo"] = coll_info_gpu["algo"].map(algo_mapping)
        coll_info_gpu["proto"] = coll_info_gpu["proto"].map(proto_mapping)
        
        coll_kernel_gpu = coll_kernels[gpu_id].copy()
        
        p2p_kernel_gpu = p2p_kernels[gpu_id].copy()
        p2p_kernel_gpu["proto"] = p2p_kernel_gpu["proto"].map(proto_mapping)
        # Vectorized bitwise operation
        hi32 = p2p_kernel_gpu["countHi32"].to_numpy(dtype=np.int64)
        lo32 = p2p_kernel_gpu["countLo32"].to_numpy(dtype=np.int64)
        p2p_kernel_gpu["count"] = (hi32 << 32) | lo32
        p2p_kernel_gpu.drop(columns=["countHi32", "countLo32"], inplace=True)
        
        comm_data_gpu = comm_data[gpu_id].copy()
        comm_data_gpu = comm_data_gpu.dropna(subset=["commId", "start", "end", "stream", "collective"])
        comm_data_gpu = comm_data_gpu.merge(communicator_ids_numeric_df, on="commId", how="left")
        if comm_data_gpu["comm_num_id"].isna().any():
            missing = comm_data_gpu.loc[comm_data_gpu["comm_num_id"].isna(), "commId"].dropna().astype(str).unique().tolist()
            logger.warning(f"Dropping events with unknown communicator IDs on GPU {gpu_id}: {missing[:10]}")
            comm_data_gpu = comm_data_gpu.loc[~comm_data_gpu["comm_num_id"].isna()].copy()
        comm_data_gpu = comm_data_gpu.sort_values(["start"])
        
        # Vectorized context_label calculation
        parallelism_vals = comm_data_gpu["parallelism"].map(context_labels).fillna(0).astype(np.int64)
        # make sure one communicator is used by only one stream only
        for commId, group in comm_data_gpu.groupby("commId"):
            stream_ids = group["stream"].unique()
            if len(stream_ids) > 1:
                logger.warning(f"Communicator {commId} used by multiple streams: {stream_ids}")
        
        # prepare identifier tag
        comm_num_ids = comm_data_gpu["comm_num_id"].to_numpy(dtype=np.int64)
        comm_data_gpu["comm_op_id"] = comm_data_gpu["collective"].map(comm_op_ids)
        comm_op_id = comm_data_gpu["comm_op_id"].to_numpy(dtype=np.int64)
        comm_seq_ids = comm_data_gpu.groupby(["commId", "comm_op_id"]).cumcount().to_numpy(dtype=np.int64)
        comm_seq_ids[comm_data_gpu["collective"].isin(["Send", "Recv"])] = 0  # reset seq_id for point-to-point ops
        comm_identifier = (np.array(hash_sequnces(comm_num_ids, comm_op_id, comm_seq_ids)) % 1000).astype(np.int64)

        comm_data_gpu["context_label"] = parallelism_vals.to_numpy() + comm_identifier * 100

        comm_data_gpu["collective"] = comm_data_gpu["collective"].map(comm_ops)
        comm_data_gpu["communicator"] = comm_data_gpu["commId"].map(communicators)

        comm_data_gpu.drop(columns=["commId", "parallelism", "comm_num_id", "comm_op_id"], inplace=True)
        
        return {
            "coll_info": coll_info_gpu,
            "coll_kernels": coll_kernel_gpu,
            "p2p_kernels": p2p_kernel_gpu,
            "comm_data": comm_data_gpu,
            "node_id": gpu.node_id,
            "pid": gpu.pid,
        }
    
    if parallel_generation:
        # Parallel generation: prepare data, then use multiprocessing for init + goal generation
        # GPUs are chunked into n_workers chunks for coordinated stream merging
        logger.info("preparing GPU data for parallel generation")
        
        # Create shared counters for progress tracking
        manager = multiprocessing.Manager()
        init_counter = manager.Value('i', 0)
        gen_counter = manager.Value('i', 0)
        counter_lock = manager.Lock()
        
        # Create shared array for tracking merge success per GPU (used for coordinated merging)
        total_gpus = len(gpu_devices)
        merge_success_array = None
        merge_barrier = None
        if merged_streams:
            # Shared array: 1 = merge success, 0 = merge failure (indexed by rank)
            merge_success_array = multiprocessing.Array('i', [0] * total_gpus)
            # Barrier to synchronize all workers after merge phase
            merge_barrier = multiprocessing.Barrier(min(n_workers, total_gpus))
        
        # Prepare all GPU data in main process - stored in dict for initializer
        gpu_data_dict = {}
        gpu_list = []  # List of (gpu_id, rank) tuples
        for gpu_id, gpu in tqdm(gpu_devices.items(), desc="Preparing GPU data"):
            gpu_data_dict[gpu_id] = prepare_gpu_data(gpu_id, gpu)
            rank = gpu_id2goal_rank[gpu.gpu_id]
            gpu_list.append((gpu_id, rank))
        
        # Chunk GPUs into n_workers chunks (one chunk per worker)
        # This ensures coordinated stream merging across all GPUs
        def chunk_list(lst, n_chunks):
            """Split list into n roughly equal chunks."""
            chunk_size = len(lst) // n_chunks
            remainder = len(lst) % n_chunks
            chunks = []
            start = 0
            for i in range(n_chunks):
                # Distribute remainder across first 'remainder' chunks
                end = start + chunk_size + (1 if i < remainder else 0)
                chunks.append(lst[start:end])
                start = end
            chunks = [chunk for chunk in chunks if chunk]  # remove empty chunks
            return chunks
        
        gpu_chunks = chunk_list(gpu_list, n_workers)
        # Build task list: each task is a chunk of GPUs
        task_list = [
            (chunk, 0, init_counter, gen_counter, counter_lock)  # nic=0
            for chunk in gpu_chunks
        ]
        
        time_finish_prep = time.time()
        logger.info(f"Data preparation time: {time_finish_prep - script_start_time:.2f} seconds")
        
        # Now run init + goal generation in parallel processes
        logger.info(f"initializing GPUs and generating goal files in parallel with {len(gpu_chunks)} workers")  ## len(gpu_chunks) = min(n_workers, total_gpus)
        logger.info(f"Total GPUs: {total_gpus}, chunked into {len(gpu_chunks)} chunks")
        if merged_streams:
            logger.info("Stream merging enabled: coordinated across all GPUs (all-or-none)")
        
        # Use initializer to set up shared data once per worker (copy-on-write friendly)
        with multiprocessing.Pool(
            processes=len(gpu_chunks),  ## len(gpu_chunks) = min(n_workers, total_gpus)
            initializer=_init_worker,
            initargs=(gpu_data_dict, gpu_id2goal_rank, output_dir, merged_streams, write_buffer_size,
                      merge_success_array, merge_barrier)
        ) as pool:
            # Start async map - each task processes a chunk of GPUs
            async_result = pool.map_async(process_gpu_chunk, task_list)
            
            # Create dual progress bars
            init_bar = tqdm(total=total_gpus, desc="Initializing ", position=0, 
                           bar_format='{desc}: {bar}| {n_fmt}/{total_fmt}',
                           colour='cyan', leave=True)
            gen_bar = tqdm(total=total_gpus, desc="Generating   ", position=1,
                          bar_format='{desc}: {bar}| {n_fmt}/{total_fmt}',
                          colour='green', leave=True)
            
            # Monitor progress until all tasks complete
            while not async_result.ready():
                # Update bars based on shared counters
                current_init = init_counter.value
                current_gen = gen_counter.value
                
                init_bar.n = current_init
                gen_bar.n = current_gen
                init_bar.refresh()
                gen_bar.refresh()
                
                time.sleep(0.1)
            
            # Final update to capture any last increments missed by the loop
            init_bar.n = init_counter.value
            gen_bar.n = gen_counter.value
            init_bar.refresh()
            gen_bar.refresh()
            init_bar.close()
            gen_bar.close()
            
            # Get results (will raise if any worker failed)
            results = async_result.get()
        
        # Log merge result
        if merged_streams and merge_success_array is not None:
            all_merged = all(merge_success_array[i] == 1 for i in range(total_gpus))
            if all_merged:
                logger.info("Stream merging: SUCCESS - all GPUs merged their streams")
            else:
                failed_ranks = [i for i in range(total_gpus) if merge_success_array[i] == 0]
                logger.info(f"Stream merging: SKIPPED - {len(failed_ranks)} GPU(s) could not merge (ranks: {failed_ranks[:10]}{'...' if len(failed_ranks) > 10 else ''})")
        
        time_finish_init = time.time()
        
        if concatenate:
            logger.info("concatenating goal files")
            goal_path = concatenate_goal_files(output_dir, len(gpu_devices), delete_parts=delete_parts,
                                               write_buffer_size=write_buffer_size)
        else:
            goal_path = output_dir
            logger.info(f"Goal files written to: {output_dir}/rank_*.goal")
    else:
        # Sequential mode: init GPUs then generate goals
        logger.info("initilizing GPU devices from dataframes")
        
        def init_one_gpu(gpu_id, gpu):
            gpu_data = prepare_gpu_data(gpu_id, gpu)
            gpu.init_from_dfs(
                gpu_data["coll_info"],
                gpu_data["coll_kernels"],
                gpu_data["p2p_kernels"],
                gpu_data["comm_data"]
            )
            return gpu_id
        
        if parallel_generation:
            from dask import delayed
            import dask
            tasks = [delayed(init_one_gpu)(gpu_id, gpu) for gpu_id, gpu in gpu_devices.items()]
            with TqdmCallback(desc="init GPUs", tqdm_class=_tqdm_for_dask):
                dask.compute(*tasks, scheduler="threads")
        else:
            for gpu_id, gpu in tqdm(gpu_devices.items()):
                init_one_gpu(gpu_id, gpu)

        time_finish_init = time.time()

        # Coordinated stream merging: all GPUs merge only if all can merge
        merged_streams_dict = {}
        if merged_streams:
            all_can_merge = True
            for gpu in gpu_devices.values():
                merged = gpu.merge_streams()
                if merged is None:
                    all_can_merge = False
                    logger.info(f"Stream merging: GPU {gpu.id} cannot merge streams")
                    break
                merged_streams_dict[gpu.gpu_id] = merged
            
            if all_can_merge:
                logger.info("Stream merging: SUCCESS - all GPUs merged their streams")
                for gpu in gpu_devices.values():
                    gpu.streams = {"stream_merged": merged_streams_dict[gpu.gpu_id]}
            else:
                logger.info("Stream merging: SKIPPED - not all GPUs could merge")

        # Sequential write: single output file
        goal_path = output_dir / "output.goal"
        buffering = write_buffer_size if write_buffer_size and write_buffer_size > 0 else -1
        with open(goal_path, "w", buffering=buffering) as f:
            logger.info("writing goal file")
            gpus = gpu_devices.values()
            f.write(f"num_ranks {len(gpus)}\n")
            for gpu in tqdm(gpus):
                f.write(f"rank {gpu_id2goal_rank[gpu.gpu_id]} {{\n")
                for line in gpu.generate_goal_lines(gpu_id2goal_rank, nic=0):
                    f.write(f"{line}\n")
                f.write("}\n")
    
    script_finish_time = time.time()
    logger.info(f"Time to initialize GPU devices: {time_finish_init - script_start_time:.2f} seconds")
    logger.info(f"Time to generate goal file: {script_finish_time - time_finish_init:.2f} seconds")
    logger.info(f"Total script time: {time.time() - script_start_time:.2f} seconds")
    if isinstance(goal_path, pathlib.Path) and goal_path.is_file():
        logger.info(f"Goal file written to: {goal_path}")
# %%
