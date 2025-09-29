from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nccl_comm import CommOp
# from nccl_comm import CommOp
from goal import GoalOp, GoalCalc, GoalSequential, GoalParallel
from typing import List, Dict, Type, Union, Optional, Tuple


class GPUStream:
    def __init__(self, self_gpu: GPUDevice, context_info: int = -1):
        self.context_info: int = context_info
        self.self_gpu: GPUDevice = self_gpu
        self.collectives: List[CommOp] = []
        self.coll_starts: List[int] = []
        self.coll_ends: List[int] = []

    def add_collective(self, coll: CommOp, start: int, end: int) -> None:
        self.collectives.append(coll)
        self.coll_starts.append(start)
        self.coll_ends.append(end)
    
    def generate_goal(self, starting_cpu_id: int, nic: int, gpu2goal_rank: Dict[GPUDevice, int], gpu2node: Dict[GPUDevice, int]) -> Tuple[GoalOp, int]:
        curr_cpu = starting_cpu_id
        last_cpu = curr_cpu
        prev_end = -1
        goal_ops = []
        for coll, start, end in zip(self.collectives, self.coll_starts, self.coll_ends):
            primitives = coll.to_primitives()
            goal_op, _last_cpu = primitives.to_goal(gpu2goal_rank, starting_cpu_id, nic, gpu2node)
            last_cpu = max(last_cpu, _last_cpu)
            if prev_end > 0:
                goal_ops.append(GoalCalc(starting_cpu_id, start - prev_end, curr_cpu))
            goal_ops.append(goal_op)
            prev_end = end
        return GoalSequential(gpu2goal_rank[self.self_gpu], starting_cpu_id, goal_ops), last_cpu

class GPUDevice:
    def __init__(self, id: int):
        self.id: int = id
        self.streams: Dict[str, GPUStream] = {}
    def __repr__(self):
        return f"GPUDevice(id={self.id})"
    def __eq__(self, value):
        if not isinstance(value, GPUDevice):
            return False
        return self.id == value.id
    def __hash__(self):
        return hash(self.id)
    def add_collective(self, stream: str, coll: CommOp, start: int, end: int, context: int = -1) -> None:
        self.streams.setdefault(stream, GPUStream(self, context)).add_collective(coll, start, end)

    def generate_goal(self, gpu2goal_rank: Dict[GPUDevice, int], gpu2node: Dict[GPUDevice, int], nic: int) -> int:
        goal_result = []
        starting_cpu_id = 0
        for stream_id, stream in self.streams.items():
            goal_op, starting_cpu_id = stream.generate_goal(starting_cpu_id, nic, gpu2goal_rank, gpu2node)
            goal_result.append(goal_op)
        return GoalParallel(gpu2goal_rank[self], 0, goal_result), starting_cpu_id