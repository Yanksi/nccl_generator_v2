import sqlite3
import re
import os
from dataclasses import dataclass, replace
from typing import List, Optional, Dict, Any, Tuple, NamedTuple
from enum import Enum
from collections import defaultdict
from tqdm import tqdm
import numba
import pandas as pd
# import modin.pandas as pd
import numpy as np
import logging
import pathlib


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
            dtype={"start": "Int64", "end": "Int64", "value": "string", "deviceId": "Int64", "streamId": "Int64", "pid": "Int64"}
        )
        conn.close()
        df_tmp["nodeId"] = node_id
        df_tmp["collective"] = df_tmp["value"].str.extract(r"ncclDevKernel_([a-zA-Z]+)")
        kernel_dfs.append(df_tmp)

    kernel_df = pd.concat(kernel_dfs, ignore_index=True)
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
            dtype={"start": "Int64", "end": "Int64", "text": "string"}
        )
        conn.close()
        df_tmp["nodeId"] = node_id
        nvtx_dfs.append(df_tmp)

    nvtx_df = pd.concat(nvtx_dfs, ignore_index=True)
    nvtx_df['eventId'] = range(len(nvtx_df))  # Add unique ID to each row
    return nvtx_df

def get_communicator_info(data: pd.DataFrame):
    logger.info("extracting communicator info")
    # extract available informations from the table
    comm_info_pattern = r"commHash (0x[0-9a-f]+) commId (0x[0-9a-f]+) rank (\d+) nranks (\d+) pid (\d+)"
    comm_info = data[data['text'].str.match(comm_info_pattern)].copy()
    comm_info[["commHash", "commId", "rank", "nRanks", "pid"]] = comm_info["text"].str.extract(comm_info_pattern)
    comm_info[["nRanks", "rank", "pid"]] = comm_info[["nRanks", "rank", "pid"]].astype("Int64")
    comm_info = comm_info.drop(columns=["text", "start", "end"])
    
    comm_hash2id = comm_info[["nodeId", "commHash", "commId"]].drop_duplicates()

    logger.info("extracting communicator rings")
    comm_ring_pattern = r"commHash (0x[0-9a-f]+) Rings \[(\d+)\] (\d+)->(\d+)->(\d+) pid (\d+)"
    comm_ring_info = data[data['text'].str.match(comm_ring_pattern)].copy()
    comm_ring_info[["commHash", "channelId", "prevRank", "myRank", "nextRank", "pid"]] = comm_ring_info["text"].str.extract(comm_ring_pattern)
    comm_ring_info[["channelId", "prevRank", "myRank", "nextRank", "pid"]] = comm_ring_info[["channelId", "prevRank", "myRank", "nextRank", "pid"]].astype("Int64")
    comm_ring_info = comm_ring_info.merge(
        comm_hash2id,
        on=["nodeId", "commHash"],
        how="left"
    ).drop(columns=["commHash", "text", "start", "end"])

    logger.info("extracting communicator trees")
    comm_tree_pattern = r"commHash (0x[0-9a-f]+) Trees \[(\d+)\] (-?\d+)/(-?\d+)/(-?\d+)->(-?\d+)->(-?\d+) pid (\d+)"
    comm_tree_info = data[data['text'].str.match(comm_tree_pattern)].copy()
    comm_tree_info[["commHash", "channelId", "child1Rank", "child2Rank", "child3Rank", "myRank", "parentRank", "pid"]] = comm_tree_info["text"].str.extract(comm_tree_pattern)
    comm_tree_info[["channelId", "child1Rank", "child2Rank", "child3Rank", "myRank", "parentRank", "pid"]] = comm_tree_info[["channelId", "child1Rank", "child2Rank", "child3Rank", "myRank", "parentRank", "pid"]].astype("Int64")
    comm_tree_info = comm_tree_info.merge(
        comm_hash2id,
        on=["nodeId", "commHash"],
        how="left"
    ).drop(columns=["commHash", "text", "start", "end"])
    return comm_info, comm_ring_info, comm_tree_info

def get_profiling_interval(data: pd.DataFrame):
    logger.info("extracting profiling intervals")
    data = data[["nodeId", "start", "text"]].copy()
    profile_start_pattern = r"nsys profiling start, pid: (\d+)"
    profile_end_pattern = r"nsys profiling stopped, pid: (\d+)"
    profile_start_info = data[data['text'].str.match(profile_start_pattern)].copy()
    profile_end_info = data[data['text'].str.match(profile_end_pattern)].copy()
    profile_start_info["pid"] = profile_start_info["text"].str.extract(profile_start_pattern).astype("Int64")
    profile_start_info = profile_start_info.drop(columns=["text"])
    profile_end_info["pid"] = profile_end_info["text"].str.extract(profile_end_pattern).astype("Int64")
    profile_end_info = profile_end_info.rename(columns={"start": "end"}).drop(columns=["text"])
    return profile_start_info.merge(
        profile_end_info,
        on=["nodeId", "pid"]
    )[["nodeId", "pid", "start", "end"]]

@numba.njit
def _associate_events(interval_starts, interval_ends, interval_id, events_time):
    associated_ids = -1 * np.ones(len(events_time), dtype=np.int64)
    j = 0
    for i in range(len(events_time)):
        while j < len(interval_starts) and interval_ends[j] < events_time[i]:
            j += 1
        if j < len(interval_starts) and interval_starts[j] <= events_time[i] < interval_ends[j]:
            associated_ids[i] = interval_id[j]
    return associated_ids

def _filter_time(profiling_interval: pd.DataFrame, data: pd.DataFrame):
        merged = data.merge(
            profiling_interval,
            on=["nodeId", "pid"],
            suffixes=("", "_profile")
        )
        return merged[
            (merged["start"] < merged["end_profile"]) & (merged["end"] > merged["start_profile"])
        ].drop(columns=["start_profile", "end_profile"])

def get_event_info(data: pd.DataFrame, profiling_interval: pd.DataFrame = None):
    logger.info("extracting event infos")
    comm_pattern = r"nccl([a-zA-Z]+)\(\): commHash (0x[0-9a-f]+), stream (0x[0-9a-f]+), data_size (\d+), type_size (\d+),.* pid (\d+)"
    comm_data = data[data['text'].str.match(comm_pattern)].copy()
    comm_data[["collective", "commHash", "stream", "data_size", "type_size", "pid"]] = comm_data["text"].str.extract(comm_pattern)
    comm_data[["data_size", "type_size", "pid"]] = comm_data[["data_size", "type_size", "pid"]].astype("Int64")

    coll_info_pattern = r'collType (\d+) root (\d+) redOp (\d+) algo (\d+) proto (\d+) commHash (\S+) stream (\S+) data_size (\d+) type_size (\d+) chunkSize (\d+) chunkCount (\d+) chunkSteps (\d+) sliceSteps (\d+) stepSize (\d+) pid (\d+)'
    coll_info_data = data[data['text'].str.match(coll_info_pattern)].copy()
    coll_info_data[["collType", "root", "redOp", "algo", "proto", "commHash", "stream", "data_size", "type_size", "chunkSize", "chunkCount", "chunkSteps", "sliceSteps", "stepSize", "pid"]] = coll_info_data["text"].str.extract(coll_info_pattern)
    coll_info_data[["collType", "root", "redOp", "algo", "proto", "data_size", "type_size", "chunkSize", "chunkCount", "chunkSteps", "sliceSteps", "stepSize", "pid"]] = coll_info_data[["collType", "root", "redOp", "algo", "proto", "data_size", "type_size", "chunkSize", "chunkCount", "chunkSteps", "sliceSteps", "stepSize", "pid"]].astype("Int64")

    coll_kernel_pattern = r'nWarps (\d+) count (\d+) chunkCount (\d+) workCount (\d+) lastChunkCount (\d+) workOffset (\d+) sendbuff (\d+) recvbuff (\d+) pid (\d+)'
    coll_kernel_data = data[data['text'].str.match(coll_kernel_pattern)].copy()
    coll_kernel_data[["nWarps", "count", "chunkCount", "workCount", "lastChunkCount", "workOffset", "sendbuff", "recvbuff", "pid"]] = coll_kernel_data["text"].str.extract(coll_kernel_pattern)
    coll_kernel_data[["nWarps", "count", "chunkCount", "workCount", "lastChunkCount", "workOffset", "sendbuff", "recvbuff", "pid"]] = coll_kernel_data[["nWarps", "count", "chunkCount", "workCount", "lastChunkCount", "workOffset", "sendbuff", "recvbuff", "pid"]].astype("Int64")

    p2p_kernel_pattern = r'Bytes (\d+) nWarps (\d+) p2pType (\d+) peer (\d+) proto (\d+) countHi32 (\d+) countLo32 (\d+) chunkSize (\d+) pid (\d+)'
    p2p_kernel_data = data[data['text'].str.match(p2p_kernel_pattern)].copy()
    p2p_kernel_data[["Bytes", "nWarps", "p2pType", "peer", "proto", "countHi32", "countLo32", "chunkSize", "pid"]] = p2p_kernel_data["text"].str.extract(p2p_kernel_pattern)
    p2p_kernel_data[["Bytes", "nWarps", "peer", "proto", "countHi32", "countLo32", "chunkSize", "pid"]] = p2p_kernel_data[["Bytes", "nWarps", "peer", "proto", "countHi32", "countLo32", "chunkSize", "pid"]].astype("Int64")
    
    comm_grouped = {name: group for name, group in comm_data.groupby(['nodeId', 'pid'])}
    coll_info_grouped = {name: group for name, group in coll_info_data.groupby(['nodeId', 'pid'])}
    coll_kernel_grouped = {name: group for name, group in coll_kernel_data.groupby(['nodeId', 'pid'])}
    p2p_kernel_grouped = {name: group for name, group in p2p_kernel_data.groupby(['nodeId', 'pid'])}

    logger.info("associating events")
    for gpu in tqdm(comm_grouped.keys()):
        comm = comm_grouped[gpu]
        comm = comm.sort_values(by="start").reset_index(drop=True)

        coll_comm = comm[(comm["collective"] != "Send") & (comm["collective"] != "Recv")]
        if len(coll_comm) > 0:
            coll_infos = coll_info_grouped[gpu]
            coll_kernels = coll_kernel_grouped[gpu]
            
            coll_infos = coll_infos.sort_values(by="start").reset_index(drop=True)
            coll_kernels = coll_kernels.sort_values(by="start").reset_index(drop=True)

            comm_starts = coll_comm["start"].to_numpy()
            comm_ends = np.concat([comm_starts[1:], np.array([np.iinfo(np.int64).max])])
            
            coll_info_starts = coll_infos["start"].to_numpy()
            coll_infos["association"] = _associate_events(comm_starts, comm_ends, coll_comm["eventId"].to_numpy(), coll_info_starts)

            coll_kernel_starts = coll_kernels["start"].to_numpy()
            coll_kernels["association"] = _associate_events(comm_starts, comm_ends, coll_comm["eventId"].to_numpy(), coll_kernel_starts)
            coll_info_grouped[gpu] = coll_infos
            coll_kernel_grouped[gpu] = coll_kernels

        p2p_comm = comm[(comm["collective"] == "Send") | (comm["collective"] == "Recv")]
        if len(p2p_comm) > 0:
            p2p_kernels = p2p_kernel_grouped[gpu]
            p2p_kernels = p2p_kernels.sort_values(by="start").reset_index(drop=True)
            comm_starts = p2p_comm["start"].to_numpy()
            comm_ends = np.concat([comm_starts[1:], np.array([np.iinfo(np.int64).max])])
            p2p_kernel_starts = p2p_kernels["start"].to_numpy()
            p2p_kernels["association"] = _associate_events(comm_starts, comm_ends, p2p_comm["eventId"].to_numpy(), p2p_kernel_starts)
            p2p_kernel_grouped[gpu] = p2p_kernels
    
    if len(coll_info_grouped) == 0:
        coll_info_grouped[("0", 0)] = pd.DataFrame(columns=list(coll_info_data.columns) + ["association"])
    if len(coll_kernel_grouped) == 0:
        coll_kernel_grouped[("0", 0)] = pd.DataFrame(columns=list(coll_kernel_data.columns) + ["association"])
    if len(p2p_kernel_grouped) == 0:
        p2p_kernel_grouped[("0", 0)] = pd.DataFrame(columns=list(p2p_kernel_data.columns) + ["association"])
    coll_info_data = pd.concat(list(coll_info_grouped.values()), ignore_index=True)
    coll_kernel_data = pd.concat(list(coll_kernel_grouped.values()), ignore_index=True)
    p2p_kernel_data = pd.concat(list(p2p_kernel_grouped.values()), ignore_index=True)
    
    # get the events grouped by stream    
    if profiling_interval is not None:
        logger.info("filtering events by profiling intervals")
        comm_data = _filter_time(profiling_interval, comm_data)
        coll_info_data = coll_info_data.merge(
            comm_data[["eventId"]],
            left_on=["association"],
            right_on=["eventId"],
            how="inner",
            suffixes=("", "_comm")
        ).drop(columns=["eventId_comm"])
        
        coll_kernel_data = coll_kernel_data.merge(
            comm_data[["eventId"]],
            left_on=["association"],
            right_on=["eventId"],
            how="inner",
            suffixes=("", "_comm")
        ).drop(columns=["eventId_comm"])
        
        p2p_kernel_data = p2p_kernel_data.merge(
            comm_data[["eventId"]],
            left_on=["association"],
            right_on=["eventId"],
            how="inner",
            suffixes=("", "_comm")
        ).drop(columns=["eventId_comm"])
    return comm_data, coll_info_data, coll_kernel_data, p2p_kernel_data


def associate_kernel_to_nvtx(comm_data: pd.DataFrame, kernel_events: pd.DataFrame, profiling_interval: pd.DataFrame = None):
    if profiling_interval is not None:
        logger.info("filtering kernel events by profiling intervals")
        kernel_events_filtered = _filter_time(kernel_events, profiling_interval)
    logger.info("associating kernels to nvtx events")
    kernel_df_grouped = {name: group for name, group in kernel_events_filtered.groupby(['nodeId', 'pid'])}
    comm_grouped = {name: group for name, group in comm_data.groupby(['nodeId', 'pid'])}
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
        kernels["label"] = kernels["collective"].map(collective_labels)
        nvtxs["label"] = nvtxs["collective"].map(collective_labels)
        
        kernel_stream_collectives = kernels.groupby(["streamId"]).agg({
            "label": lambda x: "".join(x),
        }).reset_index()
        kernel_stream_collectives["fingerPrint"] = kernel_stream_collectives["label"].map(
            lambda x: hex(hash(x) & 0xFFFFFFFFFFFFFFFF)
        )
        
        nvtx_stream_collectives = nvtxs.groupby(["stream"]).agg({
            "label": lambda x: "".join(x),
        }).reset_index()
        nvtx_stream_collectives["fingerPrint"] = nvtx_stream_collectives["label"].map(
            lambda x: hex(hash(x) & 0xFFFFFFFFFFFFFFFF)
        )
        
        stream_correspondence = kernel_stream_collectives.merge(
            nvtx_stream_collectives,
            on=["fingerPrint"],
            suffixes=("_kernel", "_nvtx"),
            how="outer"
        )
        if len(stream_correspondence) != max(len(kernel_stream_collectives), len(nvtx_stream_collectives)):
            raise ValueError("Mismatch in number of unique stream fingerprints")
        # check for unmatched streams
        unmatched_kernel_streams = stream_correspondence[stream_correspondence["stream"].isna()]
        if len(unmatched_kernel_streams) != 0:
            logger.error(f"GPU {gpu}: unmatched kernel streams: {unmatched_kernel_streams['streamId'].tolist()}")
            for row in unmatched_kernel_streams.itertuples():
                logger.error(f"  kernel streamId: {row.streamId}, label: {row.label_kernel}, fingerprint: {row.fingerPrint}")
            raise ValueError("Unmatched kernel streams found")
        
        nvtxs["inStreamEventId"] = nvtxs.groupby("stream").cumcount()
        kernels["inStreamEventId"] = kernels.groupby("streamId").cumcount()
        kernels = kernels.merge(
            stream_correspondence[["streamId", "stream"]],
            on=["streamId"],
            how="left"
        ).merge(
            nvtxs[["stream", "inStreamEventId", "eventId"]],
            on=["stream", "inStreamEventId"],
            how="left"
        ).drop(columns=["inStreamEventId", "stream", "label"]
        ).rename(columns={"eventId": "association"})
        
        kernels_list.append(kernels)
        
    kernel_events = pd.concat(kernels_list, ignore_index=True)
    logger.info("updating communicator time based on associated kernels")
    return comm_data.drop(columns=["start", "end"]).merge(
        kernel_events[["start", "end", "association"]],
        left_on=["eventId"],
        right_on=["association"]
    ).drop(columns=["association"]), kernel_events
    
def add_context_parallelism(comm_data: pd.DataFrame):
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
            if label_seq[i:i+2] != first_two:
                return False
        return True
    
    def get_rule(label_seq):
        if rule_dp(label_seq):
            return "DP"
        elif rule_pp(label_seq):
            return "PP"
        else:
            return "Other"

    comm_data = comm_data.copy()
    comm_data["label"] = comm_data["collective"].map(collective_labels)
    comm_grouped = comm_data.groupby(["nodeId", "pid", "stream"]).agg(
        label_seq=("label", lambda x: "".join(x))
    ).reset_index()
    comm_grouped["parallelism"] = comm_grouped["label_seq"].map(get_rule)
    return comm_data.merge(
        comm_grouped[["nodeId", "pid", "stream", "parallelism"]],
        on=["nodeId", "pid", "stream"],
        how="left"
    ).drop(columns=["label"])


if __name__ == "__main__":
    traces = find_all_traces("traces/Llama70B_N64_GPU256_TP1_PP8_DP32_70B_BS32/sqlite")
    kernel_events = get_kernel_events(traces)
    nvtx_events = get_nvtx_events(traces)
    comm_info, comm_ring_info, comm_tree_info = get_communicator_info(nvtx_events)
    profiling_interval = get_profiling_interval(nvtx_events)
    comm_data, coll_info, coll_kernels, p2p_kernels = get_event_info(nvtx_events, profiling_interval)
    kernel_events = associate_kernel_to_nvtx(comm_data, kernel_events, profiling_interval)
    comm_data = add_context_parallelism(comm_data)