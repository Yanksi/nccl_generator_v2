from __future__ import annotations
from typing import TYPE_CHECKING
# from nccl_comm import CommOp, CollectiveOp, P2POp
if TYPE_CHECKING:
    from nccl_comm import CommOp
from goal import GoalOp, GoalCalc, GoalSequential, GoalParallel
from typing import List, Dict, Type, Union, Optional, Tuple
from tqdm import tqdm
from functools import reduce


class GPUStream:
    def __init__(self, self_gpu: GPUDevice, context_info: int = -1):
        self.context_info: int = context_info
        self.self_gpu: GPUDevice = self_gpu
        self.collectives: List[Union[CommOp, int]] = []
        self.coll_starts: List[int] = []
        self.coll_ends: List[int] = []

    def add_collective(self, coll: Union[CommOp, int], start: int, end: int) -> None:
        self.collectives.append(coll)
        self.coll_starts.append(start)
        self.coll_ends.append(end)
    
    def construct_collective(self, event_id: int):
        from nccl_comm import P2POp, CollectiveOp
        comm_data = self.self_gpu.dfs["comm_data"].loc[event_id]
        coll_class = comm_data["collective"]
        if issubclass(coll_class, CollectiveOp):
            from nccl_comm import CollInfo, CollChnlInfo
            if event_id not in self.self_gpu.dfs["coll_info"].index:
                print(f"Event ID {event_id} not found in coll_info for GPU {self.self_gpu.id}.")
                return None
            coll_info = self.self_gpu.dfs["coll_info"].loc[event_id]
            coll_info = CollInfo(
                root_rank=coll_info["root"],
                red_op=coll_info["redOp"],
                algo=coll_info["algo"],
                proto=coll_info["proto"],
                data_size=coll_info["data_size"],
                type_size=coll_info["type_size"],
                chunk_steps=coll_info["chunkSteps"],
                slice_steps=coll_info["sliceSteps"],
                step_size=coll_info["stepSize"],
            )
            coll_chnl_infos = self.self_gpu.dfs["coll_kernels"][event_id].apply(
                lambda row: CollChnlInfo(
                    count=row["count"],
                    chunk_count=row["chunkCount"],
                    work_count=row["workCount"],
                    last_chunk_count=row["lastChunkCount"],
                    work_offset=row["workOffset"],
                    send_buff=row["sendbuff"],
                    recv_buff=row["recvbuff"],
                ),
                axis=1
            )
            return coll_class(self.self_gpu, comm_data["communicator"], coll_info, coll_chnl_infos, comm_data["context_label"])
            
        elif issubclass(coll_class, P2POp):
            from nccl_comm import P2PChnlInfo
            p2p_chnl_infos = self.self_gpu.dfs["p2p_kernels"][event_id].apply(
                lambda row: P2PChnlInfo(
                    Bytes=row["Bytes"],
                    proto=row["proto"],
                    count=row["count"],
                    chunk_size=row["chunkSize"],
                    peer_rank=row["peer"],
                ),
                axis=1
            )
            return coll_class(self.self_gpu, comm_data["communicator"], p2p_chnl_infos, comm_data["context_label"])
        else:
            raise ValueError(f"Unknown collective class {coll_class} for event ID {event_id}.")

    
    def generate_goal(self, starting_cpu_id: int, nic: int, gpu2goal_rank: Dict[GPUDevice, int]) -> Tuple[GoalOp, int]:
        curr_cpu = starting_cpu_id
        last_cpu = curr_cpu
        prev_end = -1
        goal_ops = []
        for coll, start, end in tqdm(zip(self.collectives, self.coll_starts, self.coll_ends), leave=False, total=len(self.collectives)):
            if isinstance(coll, int):
                # generate the collective
                assert self.self_gpu.dfs is not None, "DFS data must be attached to GPUDevice to generate integer collectives."
                coll = self.construct_collective(coll)
                if coll is None:
                    continue
            primitives = coll.to_primitives()
            goal_op, _last_cpu = primitives.to_goal(gpu2goal_rank, starting_cpu_id, nic)
            last_cpu = max(last_cpu, _last_cpu)
            if prev_end > 0:
                goal_ops.append(GoalCalc(start - prev_end, gpu2goal_rank[self.self_gpu], curr_cpu))
            goal_ops.append(goal_op)
            prev_end = end
        return GoalSequential(goal_ops, gpu2goal_rank[self.self_gpu], starting_cpu_id), last_cpu
    
    def generate_goal_lines(self, starting_cpu_id: int, nic: int, gpu2goal_rank: Dict[GPUDevice, int]):
        curr_cpu = starting_cpu_id
        last_cpu = curr_cpu
        def goal_gen():
            nonlocal curr_cpu, last_cpu
            prev_end = -1
            # for coll, start, end in tqdm(zip(self.collectives, self.coll_starts, self.coll_ends), leave=False, total=len(self.collectives)):
            for start, end, coll in tqdm(sorted(zip(self.coll_starts, self.coll_ends, self.collectives)), leave=False, total=len(self.collectives)):
                if isinstance(coll, int):
                    # generate the collective
                    assert self.self_gpu.dfs is not None, "DFS data must be attached to GPUDevice to generate integer collectives."
                    coll = self.construct_collective(coll)
                    if coll is None:
                        continue
                primitives = coll.to_primitives()
                goal_op, _last_cpu = primitives.to_goal(gpu2goal_rank, starting_cpu_id, nic)
                last_cpu = max(last_cpu, _last_cpu)
                if prev_end > 0:
                    yield GoalCalc(start - prev_end, gpu2goal_rank[self.self_gpu], curr_cpu)
                    # yield GoalCalc(112123, gpu2goal_rank[self.self_gpu], curr_cpu)
                yield goal_op
                prev_end = end
        goal_op = GoalSequential(goal_gen(), gpu2goal_rank[self.self_gpu], starting_cpu_id)
        for line in goal_op.generate_lines():
            yield line
        tqdm.write(f"Generated goals for stream on GPU {self.self_gpu.id} with context {self.context_info} for {len(self.collectives)} collectives.")
        return last_cpu
    
    def __len__(self):
        return len(self.collectives)

    def __repr__(self):
        return f"GPUStream(context_info={self.context_info}, self_gpu={self.self_gpu}, num_collectives={len(self.collectives)})"

class GPUDevice:
    def __init__(self, id: int, node_id: int = -1, reference_mode: bool = False):
        self.id: int = id
        self.streams: Dict[str, GPUStream] = {}
        self.node_id: int = node_id
        self.dfs = None
    
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
            "coll_kernels": {k:v.sort_values("workOffset") for k,v in coll_kernels.groupby("association")},
            "p2p_kernels": {k:v for k,v in p2p_kernels.groupby("association")},
            "comm_data": comm_data.set_index("eventId")
        }
        for event_id, rows in self.dfs["comm_data"].iterrows():
            self.add_collective(
                stream=rows["stream"],
                coll=event_id,
                start=rows["start"],
                end=rows["end"],
                context=rows["context_label"]
            )
    
    def streams_sorted(self):
        sorted_stream_keys = sorted(self.streams.keys(), key=lambda x: (self.streams[x].context_info, len(self.streams[x]), x))
        for key in sorted_stream_keys:
            yield key, self.streams[key]

    def add_collective(self, stream: str, coll: Union[CommOp, int], start: int, end: int, context: int = -1) -> None:
        self.streams.setdefault(stream, GPUStream(self, context)).add_collective(coll, start, end)

    def generate_goal(self, gpu2goal_rank: Dict[GPUDevice, int], nic: int, starting_cpu_id: int = 0) -> int:
        goal_result = []
        for stream_id, stream in self.streams_sorted():
            goal_op, starting_cpu_id = stream.generate_goal(starting_cpu_id, nic, gpu2goal_rank)
            goal_result.append(goal_op)
        return GoalParallel(goal_result, gpu2goal_rank[self], 0), starting_cpu_id
    
    def generate_goal_lines(self, gpu2goal_rank: Dict[GPUDevice, int], nic: int, starting_cpu_id: int = 0):
        for stream_id, stream in self.streams_sorted():
            starting_cpu_id = yield from stream.generate_goal_lines(starting_cpu_id, nic, gpu2goal_rank)
        return starting_cpu_id

    def merge_streams(self) -> None:
        streams_id = list(self.streams.keys())
        all_collectives = reduce(lambda a, b: a + b,
                                  [self.streams[stream].collectives for stream in streams_id])
        all_starts = reduce(lambda a, b: a + b,
                                  [self.streams[stream].coll_starts for stream in streams_id])
        all_ends = reduce(lambda a, b: a + b,
                                  [self.streams[stream].coll_ends for stream in streams_id])
        all_collectives_with_times = sorted(zip(all_starts, all_ends, all_collectives), key=lambda x: x[0])
        streams = [GPUStream(self)]
        for start, end, coll in all_collectives_with_times:
            stream_id = -1
            for i, stream in enumerate(streams):
                if len(stream) == 0 or stream.coll_ends[-1] <= start:
                    stream_id = i
                    break
            if stream_id == -1:
                stream_id = len(streams)
                streams.append(GPUStream(self))
            streams[stream_id].add_collective(coll, start, end)
        self.streams = {f"merged_stream_{i}": stream for i, stream in enumerate(streams)}