from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Type, Union, Optional, Tuple, Generator

class GoalOp(ABC):
    """
    class for modeling the dependencies between different goal objects, labeling and creating edges
    """
    def __init__(self, cpu):
        self.cpu = cpu

    @abstractmethod
    def get_start_id(self) -> int:
        pass

    @abstractmethod
    def get_end_id(self) -> int:
        pass
    
    @abstractmethod
    def generate_lines(self) -> Generator[str]:
        pass
    
    def __str__(self):
        return "\n".join(self.generate_lines())

class GoalOpAtom(GoalOp, ABC):
    task_id_for_rank: Dict[int, int] = {}
    def __init__(self, self_rank: int, cpu: int):
        super().__init__(cpu)
        self.self_rank = self_rank
        self.id: int = -1

    def get_start_id(self) -> int:
        return self.get_id()
    
    def get_end_id(self) -> int:
        return self.get_id()
    
    def get_id(self) -> None:
        if self.id < 0:
            self.id = GoalOpAtom.task_id_for_rank.setdefault(self.self_rank, 0)
            GoalOpAtom.task_id_for_rank[self.self_rank] += 1
        return self.id
    
class GoalTraffic(GoalOpAtom, ABC):
    def __init__(self, self_rank: int, peer_rank: int, size: int, cpu: int, nic: int, context: int):
        super().__init__(self_rank, cpu)
        self.context = context
        self.peer_rank = peer_rank
        self.size = size
        self.nic = nic

class GoalSend(GoalTraffic):
    send_message_id: Dict[Tuple[int, int, int], int] = {}
    def __init__(self, self_rank: int, peer_rank: int, size: int, cpu: int, nic: int, context: int):
        super().__init__(self_rank, peer_rank, size, cpu, nic, context)
        self.message_id = GoalSend.send_message_id.setdefault((self.self_rank, self.peer_rank, self.context), 0)
        GoalSend.send_message_id[(self.self_rank, self.peer_rank, self.context)] += 1

    def generate_lines(self) -> Generator[str]:
        tag = str(self.context).zfill(2) + str(self.message_id).zfill(6)
        yield f"l{self.get_id()}: send {self.size}b to {self.peer_rank} cpu {self.cpu} nic {self.nic} tag {tag}"
        
class GoalRecv(GoalTraffic):
    recv_message_id: Dict[Tuple[int, int, int], int] = {}
    def __init__(self, self_rank: int, peer_rank: int, size: int, cpu: int, nic: int, context: int):
        super().__init__(self_rank, peer_rank, size, cpu, nic, context)
        self.message_id = GoalRecv.recv_message_id.setdefault((self.self_rank, self.peer_rank, self.context), 0)
        GoalRecv.recv_message_id[(self.self_rank, self.peer_rank, self.context)] += 1
    
    def generate_lines(self) -> Generator[str]:
        tag = str(self.context).zfill(2) + str(self.message_id).zfill(6)
        yield f"l{self.get_id()}: recv {self.size}b from {self.peer_rank} cpu {self.cpu} nic {self.nic} tag {tag}"

class GoalCalc(GoalOpAtom):
    def __init__(self, self_rank: int, duration: int, cpu: int):
        super().__init__(self_rank, cpu)
        self.duration = duration
    
    def generate_lines(self) -> Generator[str]:
        yield f"l{self.get_id()}: calc {self.duration} cpu {self.cpu}"

class GoalParallel(GoalOp):
    def __init__(self, self_rank: int, cpu: int, ops: Union[list[GoalOp], Generator[GoalOp]]):
        super().__init__(cpu)
        self.ops: Union[List[GoalOp], Generator[GoalOp]] = ops
        self.single_use = isinstance(ops, Generator)
        self.consumed = False
        self.starting_op = GoalCalc(self_rank, 0, cpu)
        self.ending_op = GoalCalc(self_rank, 0, cpu)

    def add_op(self, op: GoalOp):
        if self.single_use:
            raise ValueError("Cannot add op to a GoalParallel initialized with a generator.")
        self.ops.append(op)
    
    def get_start_id(self) -> int:
        return self.starting_op.get_start_id()

    def get_end_id(self) -> int:
        return self.ending_op.get_end_id()
    
    def generate_lines(self) -> Generator[str]:
        if self.consumed:
            raise ValueError("This GoalParallel has already been consumed and it is single-use.")

        yield from self.starting_op.generate_lines()
        for op in self.ops:
            yield from op.generate_lines()
            yield f"l{op.get_start_id()} requires l{self.starting_op.get_end_id()}"
        
        yield from self.ending_op.generate_lines()
        for op in self.ops:
            yield f"l{self.ending_op.get_start_id()} requires l{op.get_end_id()}"
        
        self.consumed = True and self.single_use
        # results = "\n".join([str(op) for op in self.ops] + [str(self.starting_op), str(self.ending_op)])
        # requirements_pre = "\n".join([
        #     f"l{op.get_start_id()} requires l{self.starting_op.get_end_id()}" for op in self.ops
        # ])
        # requirements_post = "\n".join([
        #     f"l{self.ending_op.get_start_id()} requires l{op.get_end_id()}" for op in self.ops
        # ])
        # return f"{results}\n{requirements_pre}\n{requirements_post}"

class GoalSequential(GoalOp):
    def __init__(self, self_rank: int, cpu: int, ops: Union[List[GoalOp], Generator[GoalOp]]):
        super().__init__(cpu)
        self.ops: Union[List[GoalOp], Generator[GoalOp]] = ops
        self.single_use = isinstance(ops, Generator)
        self.consumed = False
        self.starting_op = None
        self.ending_op = None

    def add_op(self, op: GoalOp):
        if self.single_use:
            raise ValueError("Cannot add op to a GoalSequential initialized with a generator.")
        self.ops.append(op)
    
    def get_start_id(self) -> int:
        if self.starting_op:
            return self.starting_op.get_start_id()
        self.starting_op = self.ops[0] # should automatically raise error if ops is a generator
        return self.starting_op.get_start_id()
    
    def get_end_id(self) -> int:
        if self.ending_op:
            return self.ending_op.get_end_id()
        self.ending_op = self.ops[-1] # should automatically raise error if ops is a generator
        return self.ending_op.get_end_id()

    def generate_lines(self) -> Generator[str]:
        if self.consumed:
            raise ValueError("This GoalSequential has already been consumed and it is single-use.")
        iterator = iter(self.ops)
        self.starting_op = next(iterator)
        self.ending_op = self.starting_op
        prev_op = self.starting_op
        yield from self.starting_op.generate_lines()

        for op in iterator:
            self.ending_op = op
            yield from op.generate_lines()
            yield f"l{op.get_start_id()} requires l{prev_op.get_end_id()}"
            prev_op = op
        self.consumed = True and self.single_use
        # results = "\n".join([str(op) for op in self.ops])
        # requirements = "\n".join([
        #     f"l{self.ops[i+1].get_start_id()} requires l{self.ops[i].get_end_id()}" for i in range(len(self.ops)-1)
        # ])
        # return f"{results}\n{requirements}"