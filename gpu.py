from __future__ import annotations
from typing import TYPE_CHECKING, Tuple
# from nccl_comm import CommOp, CollectiveOp, P2POp
if TYPE_CHECKING:
    from nccl_comm import CommOp
from nccl_primitives import GpuId
from goal import GoalOp, GoalCalc, GoalSequential, GoalParallel
from typing import List, Dict, Type, Union, Optional
from tqdm import tqdm
from functools import reduce
import numba
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
class GPUStream:
    def __init__(self, self_gpu: GPUDevice, context_info: int = -1):
        self.context_info: int = context_info
        self.self_gpu: GPUDevice = self_gpu
        self.collectives: List[Union[CommOp, int]] = []
        self.coll_starts: List[int] = []
        self.coll_ends: List[int] = []
        self.group_constructed: set = set()

    def add_collective(self, coll: Union[CommOp, int], start: int, end: int) -> None:
        self.collectives.append(coll)
        self.coll_starts.append(start)
        self.coll_ends.append(end)
    
    def __construct_collective(self, event_id: int):
        from nccl_comm import P2POp, CollectiveOp
        comm_data = self.self_gpu.dfs["comm_data"].loc[event_id]
        coll_class = comm_data["collective"]
        if issubclass(coll_class, CollectiveOp):
            from nccl_comm import CollInfo, CollChnlInfo
            if event_id not in self.self_gpu.dfs["coll_info"].index:
                logger.warning(f"Event ID {event_id} not found in coll_info for GPU {self.self_gpu.id}.")
                return None
            coll_info_row = self.self_gpu.dfs["coll_info"].loc[event_id]
            coll_info = CollInfo(
                root_rank=int(coll_info_row["root"]),
                red_op=int(coll_info_row["redOp"]),
                algo=coll_info_row["algo"],
                proto=coll_info_row["proto"],
                data_size=int(coll_info_row["data_size"]),
                type_size=int(coll_info_row["type_size"]),
                chunk_steps=int(coll_info_row["chunkSteps"]),
                slice_steps=int(coll_info_row["sliceSteps"]),
                step_size=int(coll_info_row["stepSize"]),
            )
            coll_chnl_infos = self.self_gpu.dfs["coll_kernels"].get_group(event_id).apply(
                lambda row: CollChnlInfo(
                    count=int(row["count"]),
                    chunk_count=int(row["chunkCount"]),
                    work_count=int(row["workCount"]),
                    last_chunk_count=int(row["lastChunkCount"]),
                    work_offset=int(row["workOffset"]),
                    send_buff=int(row["sendbuff"]),
                    recv_buff=int(row["recvbuff"]),
                ),
                axis=1
            )
            return coll_class(self.self_gpu.gpu_id, comm_data["communicator"], coll_info, coll_chnl_infos, comm_data["context_label"])
            
        elif issubclass(coll_class, P2POp):
            from nccl_comm import P2PChnlInfo
            p2p_chnl_infos = self.self_gpu.dfs["p2p_kernels"].get_group(event_id).apply(
                lambda row: P2PChnlInfo(
                    Bytes=int(row["Bytes"]),
                    proto=row["proto"],
                    count=int(row["count"]),
                    chunk_size=int(row["chunkSize"]),
                    peer_rank=int(row["peer"]),
                ),
                axis=1
            )
            return coll_class(self.self_gpu.gpu_id, comm_data["communicator"], p2p_chnl_infos, comm_data["context_label"])
        else:
            raise ValueError(f"Unknown collective class {coll_class} for event ID {event_id}.")
    
    def construct_collective(self, event_id: int):
        if self.self_gpu.dfs["comm_data"].loc[event_id]["groupId"] < 0:
            return self.__construct_collective(event_id)
        else:
            from nccl_comm import CommGrouped
            group_id = self.self_gpu.dfs["comm_data"].loc[event_id]["groupId"]
            if group_id in self.group_constructed:
                return None
            self.group_constructed.add(group_id)
            grouped_event_ids = self.self_gpu.dfs["comm_data_grouped"].get_group(group_id)["eventId"].tolist()
            comm_ops = (self.__construct_collective(eid) for eid in grouped_event_ids)
            return CommGrouped(self.self_gpu.gpu_id, comm_ops, self.self_gpu.dfs["comm_data"].loc[event_id]["context_label"])

    
    def generate_goal(self, starting_cpu_id: int, nic: int, gpu_id2goal_rank: Dict[GpuId, int]) -> Tuple[GoalOp, int]:
        curr_cpu = starting_cpu_id
        last_cpu = curr_cpu
        prev_end = -1
        goal_ops = []
        gpu_id = self.self_gpu.gpu_id
        # for coll, start, end in tqdm(zip(self.collectives, self.coll_starts, self.coll_ends), leave=False, total=len(self.collectives)):
        for start, end, coll in sorted(zip(self.coll_starts, self.coll_ends, self.collectives)):
            if isinstance(coll, int):
                # generate the collective
                assert self.self_gpu.dfs is not None, "DFS data must be attached to GPUDevice to generate integer collectives."
                coll = self.construct_collective(coll)
                if coll is None:
                    continue
            # primitives = coll.to_primitives()
            goal_op, _last_cpu = coll.to_goal(gpu_id2goal_rank, starting_cpu_id, nic)
            last_cpu = max(last_cpu, _last_cpu)
            if prev_end > 0:
                goal_ops.append(GoalCalc(start - prev_end, gpu_id2goal_rank[gpu_id], curr_cpu))
            goal_ops.append(goal_op)
            prev_end = end
        return GoalSequential(goal_ops, gpu_id2goal_rank[gpu_id], starting_cpu_id), last_cpu
    
    def generate_goal_lines(self, starting_cpu_id: int, nic: int, gpu_id2goal_rank: Dict[GpuId, int]):
        curr_cpu = starting_cpu_id
        last_cpu = curr_cpu
        gpu_id = self.self_gpu.gpu_id
        def goal_gen():
            nonlocal curr_cpu, last_cpu
            prev_end = -1
            # for coll, start, end in tqdm(zip(self.collectives, self.coll_starts, self.coll_ends), leave=False, total=len(self.collectives)):
            # for start, end, coll in tqdm(sorted(zip(self.coll_starts, self.coll_ends, self.collectives)), leave=False, total=len(self.collectives)):
            for start, end, coll in sorted(zip(self.coll_starts, self.coll_ends, self.collectives)):
                if isinstance(coll, int):
                    # generate the collective
                    assert self.self_gpu.dfs is not None, "DFS data must be attached to GPUDevice to generate integer collectives."
                    coll = self.construct_collective(coll)
                    if coll is None:
                        continue
                # primitives = coll.to_primitives()
                goal_op, _last_cpu = coll.to_goal(gpu_id2goal_rank, starting_cpu_id, nic)
                last_cpu = max(last_cpu, _last_cpu)
                if prev_end > 0:
                    yield GoalCalc(start - prev_end, gpu_id2goal_rank[gpu_id], curr_cpu)
                    # yield GoalCalc(start - prev_end, gpu_id2goal_rank[gpu_id], 100000)
                    # yield GoalCalc(112123, gpu_id2goal_rank[gpu_id], curr_cpu)
                yield goal_op
                prev_end = end
        goal_op = GoalSequential(goal_gen(), gpu_id2goal_rank[gpu_id], starting_cpu_id)
        for line in goal_op.generate_lines():
            yield line
        # tqdm.write(f"Generated goals for stream on GPU {self.self_gpu.id} with context {self.context_info} for {len(self.collectives)} collectives.")
        return last_cpu
    
    def __len__(self):
        return len(self.collectives)

    def __repr__(self):
        return f"GPUStream(context_info={self.context_info}, self_gpu={self.self_gpu}, num_collectives={len(self.collectives)})"


@numba.jit
def check_compatible(start1, end1, start2, end2) -> bool:
    curr_idx1, curr_idx2 = 0, 0
    while curr_idx1 < len(start1) and curr_idx2 < len(start2):
        s1, e1 = start1[curr_idx1], end1[curr_idx1]
        s2, e2 = start2[curr_idx2], end2[curr_idx2]
        if e1 <= s2:
            curr_idx1 += 1
        elif e2 <= s1:
            curr_idx2 += 1
        else:
            return False
    return True

def check_mergable(stream1, stream2) -> bool:
    if len(stream1) * len(stream2) == 0:
        return True
    return check_compatible(
        sorted(stream1.coll_starts),
        sorted(stream1.coll_ends),
        sorted(stream2.coll_starts),
        sorted(stream2.coll_ends),
    )

class GPUDevice:
    def __init__(self, rank: int, node_id: str = "", pid: int = 0, reference_mode: bool = False):
        self.id: int = rank  # Sequential rank for display/goal file
        self.pid: int = pid  # Actual process ID from trace
        self.streams: Dict[str, GPUStream] = {}
        self.node_id: str = node_id
        self.dfs = None
    
    @property
    def gpu_id(self) -> GpuId:
        """Return the GPU ID as a (node_id, pid) tuple."""
        return (self.node_id, self.pid)
    
    def __repr__(self):
        return f"GPUDevice(id={self.id})"

    def __eq__(self, value):
        if not isinstance(value, GPUDevice):
            return False
        return self.id == value.id

    def __hash__(self):
        return hash(self.id)

    def init_from_dfs(self, coll_info, coll_kernels, p2p_kernels, comm_data):
        self.dfs = {
            "coll_info": coll_info.set_index("association"),
            "coll_kernels": coll_kernels.groupby("association"),
            "p2p_kernels": p2p_kernels.groupby("association"),
            "comm_data": comm_data.set_index("eventId"),
            "comm_data_grouped": comm_data.groupby("groupId")
        }
        # Use itertuples() instead of iterrows() for ~10-100x speedup
        for row in self.dfs["comm_data"].itertuples():
            self.add_collective(
                stream=row.stream,
                coll=row.Index,  # event_id is the index
                start=row.start,
                end=row.end,
                context=row.context_label
            )
    
    def streams_sorted(self):
        sorted_stream_keys = sorted(self.streams.keys(), key=lambda x: (self.streams[x].context_info, len(self.streams[x]), x))
        for key in sorted_stream_keys:
            yield key, self.streams[key]

    def add_collective(self, stream: str, coll: Union[CommOp, int], start: int, end: int, context: int = -1) -> None:
        self.streams.setdefault(stream, GPUStream(self, context)).add_collective(coll, start, end)

    def generate_goal(self, gpu_id2goal_rank: Dict[GpuId, int], nic: int, starting_cpu_id: int = 0) -> int:
        goal_result = []
        for stream_id, stream in self.streams_sorted():
            goal_op, starting_cpu_id = stream.generate_goal(starting_cpu_id, nic, gpu_id2goal_rank)
            goal_result.append(goal_op)
        return GoalParallel(goal_result, gpu_id2goal_rank[self.gpu_id], 0), starting_cpu_id
    
    def generate_goal_lines(self, gpu_id2goal_rank: Dict[GpuId, int], nic: int, starting_cpu_id: int = 0):
        for stream_id, stream in self.streams_sorted():
            starting_cpu_id = yield from stream.generate_goal_lines(starting_cpu_id, nic, gpu_id2goal_rank)
        return starting_cpu_id

    # def merge_streams(self) -> None:
    #     logger.info(f"Merging streams for GPU {self.id} with {len(self.streams)} streams.")
    #     streams_id = list(self.streams.keys())
    #     all_collectives = reduce(lambda a, b: a + b,
    #                               [self.streams[stream].collectives for stream in streams_id])
    #     all_starts = reduce(lambda a, b: a + b,
    #                               [self.streams[stream].coll_starts for stream in streams_id])
    #     all_ends = reduce(lambda a, b: a + b,
    #                               [self.streams[stream].coll_ends for stream in streams_id])
    #     all_collectives_with_times = sorted(zip(all_starts, all_ends, all_collectives), key=lambda x: x[0])
    #     streams = [GPUStream(self)]
    #     for start, end, coll in all_collectives_with_times:
    #         stream_id = -1
    #         for i, stream in enumerate(streams):
    #             if len(stream) == 0 or stream.coll_ends[-1] <= start:
    #                 stream_id = i
    #                 break
    #         if stream_id == -1:
    #             stream_id = len(streams)
    #             streams.append(GPUStream(self))
    #         streams[stream_id].add_collective(coll, start, end)
    #     self.streams = {f"merged_stream_{i}": stream for i, stream in enumerate(streams)}
    #     logger.info(f"After merging, GPU {self.id} has {len(self.streams)} streams.")
    
    def merge_streams(self) -> Optional[GPUStream]:
        # logger.info(f"Merging streams for GPU {self.id} with {len(self.streams)} streams.")
        merged_stream = GPUStream(self)
        for _, stream in self.streams_sorted():
            if check_mergable(stream, merged_stream):
                # merge
                merged_stream.collectives.extend(stream.collectives)
                merged_stream.coll_starts.extend(stream.coll_starts)
                merged_stream.coll_ends.extend(stream.coll_ends)
            else:
                return None
        return merged_stream
