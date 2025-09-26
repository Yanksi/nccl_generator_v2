from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Type, Union, Optional, Tuple
from math import ceil
from .nccl_comm import CommOp
from .goal import GoalSend, GoalRecv, GoalCalc, GoalSequential, GoalParallel, GoalOp

def intra_node_transfer_time(size: int) -> int:
    # TODO
    return 0

def reduction_time(size: int) -> int:
    # TODO
    return 0

def copy_time(size: int) -> int:
    # TODO
    return 0


class GPUStream:
    def __init__(self, context_info: int = -1):
        self.context_info: int = context_info
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
                goal_ops.append(GoalCalc(start - prev_end, curr_cpu))
            goal_ops.append(goal_op)
            prev_end = end
        return GoalSequential(*goal_ops), last_cpu

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
    
    def generate_goal(self, starting_cpu_id: int, gpu2goal_rank: Dict[GPUDevice, int], gpu2node: Dict[GPUDevice, int], nic: int) -> int:
        goal_result = []
        for stream_id, stream in self.streams.items():
            goal_op, starting_cpu_id = stream.generate_goal(starting_cpu_id, nic, gpu2goal_rank, gpu2node)
            goal_result.append(goal_op)
        return GoalParallel(*goal_result), starting_cpu_id

class NCCLPrimitiveComm(ABC):
    def __init__(self, __reduced__: bool = False):
        self.__reduced__ = __reduced__

    @abstractmethod
    def proto_ll(self) -> NCCLPrimitiveComm:
        pass
    
    @abstractmethod
    def proto_simple(self, chunk_size: int) -> NCCLPrimitiveComm:
        pass
    
    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def _to_goal(self, gpu2goal_rank: Dict[GPUDevice, int], cpu: int, nic: int, gpu2node: Dict[GPUDevice, int]) -> Tuple[GoalOp, int]:
        pass

    def to_goal(self, gpu2goal_rank: Dict[GPUDevice, int], cpu: int, nic: int,  gpu2node: Dict[GPUDevice, int]) -> Tuple[GoalOp, int]:
        if not self.__reduced__:
            raise ValueError("Non-reduced operations cannot be converted to goal operations.")
        return self._to_goal(gpu2goal_rank, cpu, nic, gpu2node)

class NCCLPrimitiveParallel(NCCLPrimitiveComm):
    def __init__(self, single_executer: bool = False):
        super().__init__(True)
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
            result.add(p.proto_simple())
        return result
    
    def _to_goal(self, gpu2goal_rank: Dict[GPUDevice, int], cpu: int, nic: int, gpu2node: Dict[GPUDevice, int]) -> Tuple[GoalOp, int]:
        curr_cpu_start = cpu
        if self.single_executer:
            ops = [p.to_goal(gpu2goal_rank, curr_cpu_start, nic, gpu2node) for p in self.primitives]
            max_cpu = max(op[1] for op in ops)
            return GoalParallel(*(op[0] for op in ops)), max_cpu
        else:
            ops = []
            for p in self.primitives:
                op, curr_cpu_end = p.to_goal(gpu2goal_rank, curr_cpu_start, nic, gpu2node)
                ops.append(op)
                curr_cpu_start = curr_cpu_end
            return GoalParallel(*ops), curr_cpu_start
    
    def __repr__(self) -> str:
        return "NCCLPrimitiveParallel(" + ", ".join(repr(p) for p in self.primitives) + ")"

class NCCLPrimitiveSequantial(NCCLPrimitiveComm):
    def __init__(self):
        super().__init__(True)
        self.primitives: List[NCCLPrimitiveComm] = []
    
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
            result.append(p.proto_simple())
        return result
    
    def _to_goal(self, gpu2goal_rank: Dict[GPUDevice, int], cpu: int, nic: int, gpu2node: Dict[GPUDevice, int]) -> Tuple[GoalOp, int]:
        ops = [p.to_goal(gpu2goal_rank, cpu, nic, gpu2node) for p in self.primitives]
        max_cpu = max(op[1] for op in ops)
        return GoalSequential(*(op[0] for op in ops)), max_cpu
    
    def __repr__(self) -> str:
        return "NCCLPrimitiveSequantial(" + ", ".join(repr(p) for p in self.primitives) + ")"

class NCCLPrimitive(NCCLPrimitiveComm, ABC):
    """Base class for NCCL primitives."""
    
    def __init__(self, context: int, gpu: GPUDevice, *, source_gpu: Optional[GPUDevice] = None, 
                 target_gpu: Optional[GPUDevice] = None, size: int = 0, chunk_size: int = 0, __reduced__=False):
        super().__init__(__reduced__)
        self.context = context
        self.gpu = gpu
        self.source_gpu = source_gpu
        self.target_gpu = target_gpu
        self.size = size
        self.chunk_size = chunk_size if chunk_size > 0 else size

    def proto_ll(self) -> NCCLPrimitiveComm:
        """Convert to Low-Latency protocol primitives."""
        if self.__reduced__:
            raise ValueError("Reduced operations cannot be converted to LL protocol.")
        n_packets = ceil(self.size / 8)
        return self.__class__(
            self.gpu,
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
                self.gpu,
                source_gpu=self.source_gpu, 
                target_gpu=self.target_gpu, 
                size=self.chunk_size,
                __reduced__=True
            ))

        remaining_size = self.size % self.chunk_size
        if remaining_size > 0:
            result.add(self.__class__(
                self.gpu,
                source_gpu=self.source_gpu, 
                target_gpu=self.target_gpu, 
                 size=remaining_size,
                __reduced__=True
            ))
        return result
    
    def send_goal(self, target_goal_rank: int, size: int, cpu: int, nic: int, intra_node: bool) -> GoalOp:
        if intra_node:
            return GoalSequential(
                GoalCalc(intra_node_transfer_time(size), cpu),
                GoalSend(target_goal_rank, 0, cpu, nic, self.context)
            )
        else:
            return GoalSend(target_goal_rank, size, cpu, nic, self.context)
    
    def recv_goal(self, source_goal_rank: int, size: int, cpu: int, nic: int, intra_node: bool) -> GoalOp:
        if intra_node:
            return GoalSequential(
                GoalRecv(source_goal_rank, 0, cpu, nic, self.context),
                GoalCalc(intra_node_transfer_time(size), cpu)
            )
        else:
            return GoalRecv(source_goal_rank, size, cpu, nic, self.context)
    
    @abstractmethod
    def _p_to_goal(self, gpu2goal_rank: Dict[GPUDevice, int], cpu: int, nic: int, intra_node_send: bool, intra_node_recv: bool) -> GoalOp:
        pass

    def _to_goal(self, gpu2goal_rank: Dict[GPUDevice, int], cpu: int, nic: int, gpu2node: Dict[GPUDevice, int]) -> Tuple[GoalOp, int]:
        # check whether the send and recv are intra-node
        intra_send = gpu2node[self.gpu] == gpu2node[self.target_gpu] if self.target_gpu else False
        intra_recv = gpu2node[self.gpu] == gpu2node[self.source_gpu] if self.source_gpu else False
        return self._p_to_goal(gpu2goal_rank, cpu, nic, intra_send, intra_recv), cpu + 1


class NCCLSend(NCCLPrimitive):
    def __repr__(self) -> str:
        return f"NCCLSend(target_gpu={self.target_gpu}, size={self.size})"
    
    def _p_to_goal(self, gpu2goal_rank: Dict[GPUDevice, int], cpu: int, nic: int, intra_node_send: bool, intra_node_recv: bool) -> GoalOp:
        return self.send_goal(gpu2goal_rank[self.target_gpu], cpu, nic, intra_node_send)

    
class NCCLCopySend(NCCLPrimitive):
    def __repr__(self) -> str:
        return f"NCCLCopySend(target_gpu={self.target_gpu}, size={self.size})"

    def _p_to_goal(self, gpu2goal_rank: Dict[GPUDevice, int], cpu: int, nic: int, intra_node_send: bool, intra_node_recv: bool) -> GoalOp:
        send = self.send_goal(gpu2goal_rank[self.target_gpu], self.size, cpu, nic, intra_node_send)
        return GoalSequential(
            GoalCalc(copy_time(self.size), cpu),
            send
        )


class NCCLRecv(NCCLPrimitive):
    def __repr__(self) -> str:
        return f"NCCLRecv(source_gpu={self.source_gpu}, size={self.size})"
    
    def _p_to_goal(self, gpu2goal_rank: Dict[GPUDevice, int], cpu: int, nic: int, intra_node_send: bool, intra_node_recv: bool) -> GoalOp:
        return self.recv_goal(gpu2goal_rank[self.source_gpu], self.size, cpu, nic, intra_node_recv)

class NCCLRecvReduce(NCCLPrimitive):
    def __repr__(self) -> str:
        return f"NCCLRecvReduce(source_gpu={self.source_gpu}, size={self.size})"
    
    def _p_to_goal(self, gpu2goal_rank: Dict[GPUDevice, int], cpu: int, nic: int, intra_node_send: bool, intra_node_recv: bool) -> GoalOp:
        recv = self.recv_goal(gpu2goal_rank[self.source_gpu], self.size, cpu, nic, intra_node_recv)
        return GoalSequential(
            recv,
            GoalCalc(reduction_time(self.size), cpu)
        )

class NCCLRecvReduceCopy(NCCLPrimitive):
    def __repr__(self) -> str:
        return f"NCCLRecvReduceCopy(source_gpu={self.source_gpu}, size={self.size})"
    
    def _p_to_goal(self, gpu2goal_rank: Dict[GPUDevice, int], cpu: int, nic: int, intra_node_send: bool, intra_node_recv: bool) -> GoalOp:
        recv = self.recv_goal(gpu2goal_rank[self.source_gpu], self.size, cpu, nic, intra_node_recv)
        return GoalSequential(
            recv,
            GoalCalc(reduction_time(self.size) + copy_time(self.size), cpu)
        )

class NCCLRecvCopySend(NCCLPrimitive):
    def __repr__(self) -> str:
        return f"NCCLRecvCopySend(source_gpu={self.source_gpu}, target_gpu={self.target_gpu}, size={self.size})"
    
    def _p_to_goal(self, gpu2goal_rank: Dict[GPUDevice, int], cpu: int, nic: int, intra_node_send: bool, intra_node_recv: bool) -> GoalOp:
        recv = self.recv_goal(gpu2goal_rank[self.source_gpu], self.size, cpu, nic, intra_node_recv)
        send = self.send_goal(gpu2goal_rank[self.target_gpu], self.size, cpu, nic, intra_node_send)
        return GoalSequential(
            recv,
            GoalCalc(copy_time(self.size), cpu),
            send
        )

class NCCLRecvReduceSend(NCCLPrimitive):
    def __repr__(self):
        return f"NCCLRecvReduceSend(source_gpu={self.source_gpu.id}, target_gpu={self.target_gpu.id}, size={self.size})"
    
    def _p_to_goal(self, gpu2goal_rank: Dict[GPUDevice, int], cpu: int, nic: int, intra_node_send: bool, intra_node_recv: bool) -> GoalOp:
        recv = self.recv_goal(gpu2goal_rank[self.source_gpu], self.size, cpu, nic, intra_node_recv)
        send = self.send_goal(gpu2goal_rank[self.target_gpu], self.size, cpu, nic, intra_node_send)
        return GoalSequential(
            recv,
            GoalCalc(reduction_time(self.size), cpu),
            send
        )

class NCCLRecvReduceCopySend(NCCLPrimitive):
    def __repr__(self):
        return f"NCCLRecvReduceCopySend(source_gpu={self.source_gpu.id}, target_gpu={self.target_gpu.id}, size={self.size})"
    
    def _p_to_goal(self, gpu2goal_rank: Dict[GPUDevice, int], cpu: int, nic: int, intra_node_send: bool, intra_node_recv: bool) -> GoalOp:
        recv = self.recv_goal(gpu2goal_rank[self.source_gpu], self.size, cpu, nic, intra_node_recv)
        send = self.send_goal(gpu2goal_rank[self.target_gpu], self.size, cpu, nic, intra_node_send)
        return GoalSequential(
            recv,
            GoalCalc(reduction_time(self.size) + copy_time(self.size), cpu),
            send
        )
