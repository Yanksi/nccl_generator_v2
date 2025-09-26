from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Type, Union, Optional, Tuple

class GoalOp(ABC):
    """
    class for modeling the dependencies between different goal objects, labeling and creating edges
    """
    @abstractmethod
    def get_start_id(self) -> int:
        pass

    @abstractmethod
    def get_end_id(self) -> int:
        pass
    
    @abstractmethod
    def __str__(self):
        pass

class GoalOpAtom(GoalOp, ABC):
    task_id_for_rank: Dict[int, int] = {}
    def __init__(self, self_rank: int):
        self.id = GoalOpAtom.task_id_for_rank.setdefault(self_rank, 0)
        self_rank: int = self_rank
        GoalOpAtom.task_id_for_rank[self_rank] += 1 # each task can take two ids (mostly for the parallel case)

    def get_start_id(self) -> int:
        return self.id
    
    def get_end_id(self) -> int:
        return self.id
    
class GoalTraffic(GoalOpAtom, ABC):
    def __init__(self, self_rank: int, peer_rank: int, size: int, cpu: int, nic: int, context: int):
        super().__init__(self_rank)
        self.context = context
        self.peer_rank = peer_rank
        self.size = size
        self.cpu = cpu
        self.nic = nic

class GoalSend(GoalTraffic):
    send_message_id: Dict[Tuple[int, int], int] = {}
    def __str__(self):
        if not hasattr(self, 'message_id'):
            self.message_id = GoalSend.send_message_id.setdefault((self.peer_rank, self.size), 0)
            GoalSend.send_message_id[(self.peer_rank, self.size)] += 1
        tag = str(self.context).zfill(2) + str(self.message_id).zfill(5)
        return f"l{self.id}: send {self.size}b to {self.peer_rank} cpu {self.cpu} nic {self.nic} tag {tag}"

class GoalRecv(GoalTraffic):
    recv_message_id: Dict[Tuple[int, int], int] = {}
    def __str__(self):
        if not hasattr(self, 'message_id'):
            self.message_id = GoalRecv.recv_message_id.setdefault((self.peer_rank, self.size), 0)
            GoalRecv.recv_message_id[(self.peer_rank, self.size)] += 1
        tag = str(self.context).zfill(2) + str(self.message_id).zfill(5)
        return f"l{self.id}: recv {self.size}b from {self.peer_rank} cpu {self.cpu} nic {self.nic} tag {tag}"
    
class GoalCalc(GoalOpAtom):
    def __init__(self, self_rank: int, duration: int, cpu: int):
        super().__init__(self_rank)
        self.duration = duration
        self.cpu = cpu
    
    def __str__(self):
        return f"l{self.id}: calc {self.duration} cpu {self.cpu}"

class GoalParallel(GoalOp):
    def __init__(self, self_rank: int, *ops: GoalOp):
        self.ops: List[GoalOp] = list(ops)
        self.starting_op = GoalCalc(self_rank, 0, 0)
        self.ending_op = GoalCalc(self_rank, 0, 0)
    
    def add_op(self, op: GoalOp):
        self.ops.append(op)
    
    def get_start_id(self) -> int:
        return self.starting_op.get_start_id()

    def get_end_id(self) -> int:
        return self.ending_op.get_end_id()
    
    def __str__(self):
        results = "\n".join([str(op) for op in self.ops] + [str(self.starting_op), str(self.ending_op)])
        requirements_pre = "\n".join([
            f"l{op.get_start_id()} requires l{self.starting_op.get_end_id()}" for op in self.ops
        ])
        requirements_post = "\n".join([
            f"l{self.ending_op.get_start_id()} requires l{op.get_end_id()}" for op in self.ops
        ])
        return f"{results}\n{requirements_pre}\n{requirements_post}"

class GoalSequential(GoalOp):
    def __init__(self, self_rank: int, *ops: GoalOp):
        self.ops: List[GoalOp] = list(ops)
    
    def add_op(self, op: GoalOp):
        self.ops.append(op)
    
    def get_start_id(self) -> int:
        return self.ops[0].get_start_id() if self.ops else -1
    
    def get_end_id(self) -> int:
        return self.ops[-1].get_end_id() if self.ops else -1
    
    def __str__(self):
        results = "\n".join([str(op) for op in self.ops])
        requirements = "\n".join([
            f"l{self.ops[i+1].get_start_id()} requires l{self.ops[i].get_end_id()}" for i in range(len(self.ops)-1)
        ])
        return f"{results}\n{requirements}"