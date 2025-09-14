from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Type, Union, Optional
from math import ceil
from .nccl_comm import CommOp


def intra_node_transfer_time(size: int) -> int:
    # TODO
    return 0

def reduction_time(size: int) -> int:
    # TODO
    return 0


class GoalOp(ABC):
    """
    class for modeling the dependencies between different goal objects, labeling and creating edges
    """
    @abstractmethod
    def __repr__(self):
        pass

class GoalSend(GoalOp):
    def __init__(self, target_rank: int, size: int, tag: int, cpu: int, nic: int):
        self.target_rank = target_rank
        self.size = size
        self.tag = tag
        self.cpu = cpu
        self.nic = nic
    
    def __repr__(self):
        return f"send {self.size}b to {self.target_rank} tag {self.tag} cpu {self.cpu} nic {self.nic}"

class GoalRecv(GoalOp):
    def __init__(self, source_rank: int, size: int, tag: int, cpu: int, nic: int):
        self.source_rank = source_rank
        self.size = size
        self.tag = tag
        self.cpu = cpu
        self.nic = nic
    
    def __repr__(self):
        return f"recv {self.size}b from {self.source_rank} tag {self.tag} cpu {self.cpu} nic {self.nic}"
    
class GoalCalc(GoalOp):
    def __init__(self, duration: int, cpu: int):
        self.duration = duration
        self.cpu = cpu
    
    def __repr__(self):
        return f"calc {self.duration} cpu {self.cpu}"

class GoalParallel(GoalOp):
    def __init__(self):
        self.ops: List[GoalOp] = []
    
    def add_op(self, op: GoalOp):
        self.ops.append(op)

class GoalSequential(GoalOp):
    def __init__(self):
        self.ops: List[GoalOp] = []
    
    def add_op(self, op: GoalOp):
        self.ops.append(op)

class GPUStream:
    
    def __init__(self, context_info: int = -1):
        self.context_info: int = -1
        self.collectives: List[CommOp] = []
        self.coll_starts: List[int] = []
        self.coll_ends: List[int] = []
    def add_collective(self, coll: CommOp, start: int, end: int) -> None:
        self.collectives.append(coll)
        self.coll_starts.append(start)
        self.coll_ends.append(end)

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
        self.streams.setdefault(stream, GPUStream(context)).add_collective(coll, start, end)

class NCCLPrimitiveComm(ABC):    
    @abstractmethod
    def proto_ll(self) -> NCCLPrimitiveComm:
        pass
    
    @abstractmethod
    def proto_simple(self, chunk_size: int) -> NCCLPrimitiveComm:
        pass
    
    @abstractmethod
    def __repr__(self) -> str:
        pass

class NCCLPrimitiveParallel(NCCLPrimitiveComm):
    def __init__(self, single_executer: bool = False):
        self.single_executer = single_executer
        self.primitives: List[NCCLPrimitiveComm] = []

    def add(self, primitive: Union[NCCLPrimitiveComm]) -> None:
        self.primitives.append(primitive)
    
    def proto_ll(self) -> NCCLPrimitiveComm:
        result = NCCLPrimitiveParallel(self.single_executer)
        for p in self.primitives:
            result.add(p.proto_ll())
        return result

    def proto_simple(self) -> NCCLPrimitiveComm:
        result = NCCLPrimitiveParallel(self.single_executer)
        for p in self.primitives:
            result.add(p.proto_simple(p.chunk_size))
        return result
    
    def __repr__(self) -> str:
        return "NCCLPrimitiveParallel(" + ", ".join(repr(p) for p in self.primitives) + ")"

class NCCLPrimitiveSequantial(NCCLPrimitiveComm):
    def append(self, primitive: Union[NCCLPrimitiveComm]) -> None:
        self.primitives.append(primitive)
    
    def proto_ll(self) -> NCCLPrimitiveComm:
        result = NCCLPrimitiveSequantial()
        for p in self.primitives:
            result.append(p.proto_ll())
        return result
    
    def proto_simple(self) -> NCCLPrimitiveComm:
        result = NCCLPrimitiveSequantial()
        for p in self.primitives:
            result.append(p.proto_simple(p.chunk_size))
        return result
    
    def __repr__(self) -> str:
        return "NCCLPrimitiveSequantial(" + ", ".join(repr(p) for p in self.primitives) + ")"

class NCCLPrimitive(NCCLPrimitiveComm):
    """Base class for NCCL primitives."""
    
    def __init__(self, *, source_gpu: Optional[GPUDevice] = None, 
                 target_gpu: Optional[GPUDevice] = None, size: int = 0, chunk_size: int = 0, __reduced__=False):
        self.source_gpu = source_gpu
        self.target_gpu = target_gpu
        self.size = size
        self.chunk_size = chunk_size if chunk_size > 0 else size
        self.__reduced__ = __reduced__

    def proto_ll(self) -> NCCLPrimitiveComm:
        """Convert to Low-Latency protocol primitives."""
        if self.__reduced__:
            raise ValueError("Reduced operations cannot be converted to LL protocol.")
        n_packets = ceil(self.size / 8)
        return self.__class__(
            source_gpu=self.source_gpu,
            target_gpu=self.target_gpu,
            size=n_packets * 8,
            __reduced__=True
        )
    
    def proto_simple(self) -> NCCLPrimitiveComm:
        """Convert to Simple protocol primitives."""
        if self.__reduced__:
            raise ValueError("Reduced operations cannot be converted to Simple protocol.")
        n_chunks = self.size // self.chunk_size
        result = NCCLPrimitiveParallel(single_executer=True)
        for _ in range(n_chunks):
            result.add(self.__class__(
                source_gpu=self.source_gpu, 
                target_gpu=self.target_gpu, 
                size=self.chunk_size,
                __reduced__=True
            ))

        remaining_size = self.size % self.chunk_size
        if remaining_size > 0:
            result.add(self.__class__(
                source_gpu=self.source_gpu, 
                target_gpu=self.target_gpu, 
                 size=remaining_size,
                __reduced__=True
            ))
        return result
    
    def to_goal(self, gpu2goal_rank: Dict[GPUDevice, int], tag: int, cpu: int, nic: int, intra_node: bool):
        raise NotImplementedError("to_goal method must be implemented in subclasses.")

class NCCLSend(NCCLPrimitive):
    def __repr__(self) -> str:
        return f"NCCLSend(target_gpu={self.target_gpu}, size={self.size})"
    
    def to_goal(self, gpu2goal_rank: Dict[GPUDevice, int], tag: int, cpu: int, nic: int, intra_node: bool):
        if intra_node:
            pass

    
class NCCLCopySend(NCCLPrimitive):
    def __repr__(self) -> str:
        return f"NCCLCopySend(target_gpu={self.target_gpu}, size={self.size})"

class NCCLRecv(NCCLPrimitive):
    def __repr__(self) -> str:
        return f"NCCLRecv(source_gpu={self.source_gpu}, size={self.size})"

class NCCLRecvReduce(NCCLPrimitive):
    def __repr__(self) -> str:
        return f"NCCLRecvReduce(source_gpu={self.source_gpu}, size={self.size})"

class NCCLRecvReduceCopy(NCCLPrimitive):
    def __repr__(self) -> str:
        return f"NCCLRecvReduceCopy(source_gpu={self.source_gpu}, size={self.size})"

class NCCLRecvCopySend(NCCLPrimitive):
    def __repr__(self) -> str:
        return f"NCCLRecvCopySend(source_gpu={self.source_gpu}, target_gpu={self.target_gpu}, size={self.size})"

class NCCLRecvReduceSend(NCCLPrimitive):
    def __repr__(self):
        return f"NCCLRecvReduceSend(source_gpu={self.source_gpu.id}, target_gpu={self.target_gpu.id}, size={self.size})"

class NCCLRecvReduceCopySend(NCCLPrimitive):
    def __repr__(self):
        return f"NCCLRecvReduceCopySend(source_gpu={self.source_gpu.id}, target_gpu={self.target_gpu.id}, size={self.size})"
