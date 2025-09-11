from abc import ABC, abstractmethod
from __future__ import annotations
from typing import List, Dict, Type, Union, Optional
from math import ceil

class GPUStream:
    context_info: int = -1

class GPUDevice:
    id: any = None
    streams: Dict[str, GPUStream] = None
    def __init__(self, id: int):
        self.id = id
    def __repr__(self):
        return f"GPUDevice(id={self.id})"
    def __eq__(self, value):
        if not isinstance(value, GPUDevice):
            return False
        return self.id == value.id

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
    primitives: List[NCCLPrimitiveComm] = []
    single_executer: bool = False

    def __init__(self, single_executer: bool = False):
        self.single_executer = single_executer

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
    primitives: List[Union[NCCLPrimitiveComm]] = []
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

class NCCLSend(NCCLPrimitive):
    def __repr__(self) -> str:
        return f"NCCLSend(target_gpu={self.target_gpu}, size={self.size})"
    
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
