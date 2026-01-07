from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from functools import lru_cache
from math import ceil
from typing import Dict, Generator, List, Optional, Tuple, Type, Union

import numpy as np
from scipy import interpolate

from goal import GoalCalc, GoalOp, GoalParallel, GoalRecv, GoalSend, GoalSequential
from gpu import GPUDevice

zero_price_reduction_copy = False
zero_price_communication = False
enable_intra_node_transfer = True


def init_generation_flags(
    zero_price_reduction_copy_flag: bool,
    zero_price_communication_flag: bool,
    enable_intra_node_transfer_flag: bool,
) -> None:
    global \
        zero_price_reduction_copy, \
        zero_price_communication, \
        enable_intra_node_transfer
    zero_price_reduction_copy = zero_price_reduction_copy_flag
    zero_price_communication = zero_price_communication_flag
    enable_intra_node_transfer = enable_intra_node_transfer_flag


def intra_node_transfer_time(size: int) -> int:
    bw = 150
    return size * 10**9 // (bw * 10**9 * 1 * 2)


_DATA_CACHE = None


def init_data(simple_file, ll_file):
    """
    Load JSON files into the global variable _DATA_CACHE.
    """
    global _DATA_CACHE

    with open(simple_file, "r") as f_simple:
        data_simple = json.load(f_simple)

    with open(ll_file, "r") as f_ll:
        data_ll = json.load(f_ll)

    _DATA_CACHE = {
        2: data_simple,  # Simple protocol
        0: data_ll,  # LL protocol
    }


@lru_cache(maxsize=2048)
def reduction_time(data_size: int, proto: int) -> int:
    if zero_price_reduction_copy:
        return 0
    data = _DATA_CACHE[proto]

    if str(data_size) in data["NPKIT_EVENT_GPU_RECV_REDUCE_SEND"]:
        reduction_times = data["NPKIT_EVENT_GPU_RECV_REDUCE_SEND"][str(data_size)]
        return random.choice(reduction_times)

    # Use interpolation if the data_size is not directly in the JSON
    sizes = sorted(
        int(size) for size in data["NPKIT_EVENT_GPU_RECV_REDUCE_SEND"].keys()
    )

    if data_size < sizes[0]:
        return random.choice(data["NPKIT_EVENT_GPU_RECV_REDUCE_SEND"][str(sizes[0])])
    if data_size > sizes[-1]:
        return random.choice(data["NPKIT_EVENT_GPU_RECV_REDUCE_SEND"][str(sizes[-1])])

    f = interpolate.interp1d(
        sizes,
        [
            np.mean(data["NPKIT_EVENT_GPU_RECV_REDUCE_SEND"][str(size)])
            for size in sizes
        ],
        kind="linear",
        fill_value="extrapolate",
    )
    interpolated_value = f(data_size)

    return int(random.gauss(interpolated_value, interpolated_value * 0.01))


@lru_cache(maxsize=2048)
def copy_time(data_size: int, proto: int) -> int:
    if zero_price_reduction_copy:
        return 0
    data = _DATA_CACHE[proto]

    if str(data_size) in data["NPKIT_EVENT_GPU_DIRECT_RECV_COPY_SEND"]:
        copy_times = data["NPKIT_EVENT_GPU_DIRECT_RECV_COPY_SEND"][str(data_size)]
        return random.choice(copy_times)

    # Use interpolation if the data_size is not directly in the JSON
    sizes = sorted(
        int(size) for size in data["NPKIT_EVENT_GPU_DIRECT_RECV_COPY_SEND"].keys()
    )

    if data_size < sizes[0]:
        return random.choice(
            data["NPKIT_EVENT_GPU_DIRECT_RECV_COPY_SEND"][str(sizes[0])]
        )
    if data_size > sizes[-1]:
        return random.choice(
            data["NPKIT_EVENT_GPU_DIRECT_RECV_COPY_SEND"][str(sizes[-1])]
        )

    f = interpolate.interp1d(
        sizes,
        [
            np.mean(data["NPKIT_EVENT_GPU_DIRECT_RECV_COPY_SEND"][str(size)])
            for size in sizes
        ],
        kind="linear",
        fill_value="extrapolate",
    )
    interpolated_value = f(data_size)

    return int(random.gauss(interpolated_value, interpolated_value * 0.01))


class NCCLPrimitiveComm(ABC):
    def __init__(self, gpu: GPUDevice):
        self.gpu = gpu

    @abstractmethod
    def proto_ll(self) -> NCCLPrimitiveComm:
        pass

    @abstractmethod
    def proto_simple(self) -> NCCLPrimitiveComm:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def _to_goal(
        self, gpu2goal_rank: Dict[GPUDevice, int], cpu: int, nic: int
    ) -> Tuple[GoalOp, int]:
        pass

    def to_goal(
        self, gpu2goal_rank: Dict[GPUDevice, int], cpu: int, nic: int
    ) -> Tuple[GoalOp, int]:
        if not hasattr(self, "goal_cache"):
            self.goal_cache = self._to_goal(gpu2goal_rank, cpu, nic)
        return self.goal_cache


class NCCLPrimitiveParallel(NCCLPrimitiveComm):
    def __init__(
        self,
        gpu: GPUDevice,
        single_executer: bool = False,
        primitives: Optional[Generator[NCCLPrimitiveComm]] = None,
    ):
        super().__init__(gpu)
        self.single_executer = single_executer
        self.single_use = primitives is not None
        self.consumed = False
        self.primitives: Union[
            List[NCCLPrimitiveComm], Generator[NCCLPrimitiveComm]
        ] = primitives if primitives is not None else []

    def add(self, primitive: NCCLPrimitiveComm) -> None:
        if self.single_use:
            raise ValueError(
                "Cannot add primitive to a NCCLPrimitiveParallel initialized with a generator."
            )
        self.primitives.append(primitive)

    def proto_ll(self) -> NCCLPrimitiveComm:
        if self.consumed:
            raise ValueError(
                "This NCCLPrimitiveParallel has already been consumed and it is single-use."
            )
        result = NCCLPrimitiveParallel(
            self.gpu, self.single_executer, (p.proto_ll() for p in self.primitives)
        )
        self.consumed = True and self.single_use
        return result

    def proto_simple(self) -> NCCLPrimitiveComm:
        if self.consumed:
            raise ValueError(
                "This NCCLPrimitiveParallel has already been consumed and it is single-use."
            )
        result = NCCLPrimitiveParallel(
            self.gpu, self.single_executer, (p.proto_simple() for p in self.primitives)
        )
        self.consumed = True and self.single_use
        return result

    def _to_goal(
        self, gpu2goal_rank: Dict[GPUDevice, int], cpu: int, nic: int
    ) -> Tuple[GoalOp, int]:
        if self.consumed:
            raise ValueError(
                "This NCCLPrimitiveParallel has already been consumed and it is single-use."
            )
        self.consumed = True and self.single_use
        if self.single_executer:
            ops = [p.to_goal(gpu2goal_rank, cpu, nic) for p in self.primitives]
            max_cpu = max(op[1] for op in ops)
            return GoalParallel(
                list(op[0] for op in ops), gpu2goal_rank[self.gpu], cpu
            ), max_cpu
        else:
            ops = []
            curr_cpu_start = cpu
            for p in self.primitives:
                op, curr_cpu_start = p.to_goal(gpu2goal_rank, curr_cpu_start, nic)
                ops.append(op)
            return GoalParallel(ops, gpu2goal_rank[self.gpu], cpu), curr_cpu_start

    def __repr__(self) -> str:
        indent = "  "
        if not self.single_use:
            primitives_repr = [repr(p) for p in self.primitives]
            for i in range(len(primitives_repr)):
                primitives_repr[i] = indent + primitives_repr[i].replace(
                    "\n", "\n" + indent
                )
            return (
                f"NCCLPrimitiveParallel(single_executer={self.single_executer}\n"
                + ",\n".join(primitives_repr)
                + ")"
            )
        else:
            return f"NCCLPrimitiveParallel(single_executer={self.single_executer}, single_use=True)"


class NCCLPrimitiveSequential(NCCLPrimitiveComm):
    def __init__(self, gpu: GPUDevice, primitives: Generator[NCCLPrimitiveComm] = None):
        super().__init__(gpu)
        self.single_use = primitives is not None
        self.consumed = False
        self.primitives: Union[
            List[NCCLPrimitiveComm], Generator[NCCLPrimitiveComm]
        ] = primitives if primitives is not None else []

    def append(self, primitive: NCCLPrimitiveComm) -> None:
        if self.single_use:
            raise ValueError(
                "Cannot add primitive to a NCCLPrimitiveSequential initialized with a generator."
            )
        self.primitives.append(primitive)

    def proto_ll(self) -> NCCLPrimitiveComm:
        if self.consumed:
            raise ValueError(
                "This NCCLPrimitiveSequential has already been consumed and it is single-use."
            )
        result = NCCLPrimitiveSequential(
            self.gpu, (p.proto_ll() for p in self.primitives)
        )
        self.consumed = True and self.single_use
        return result

    def proto_simple(self) -> NCCLPrimitiveComm:
        if self.consumed:
            raise ValueError(
                "This NCCLPrimitiveSequential has already been consumed and it is single-use."
            )
        result = NCCLPrimitiveSequential(
            self.gpu, (p.proto_simple() for p in self.primitives)
        )
        self.consumed = True and self.single_use
        return result

    def _to_goal(
        self, gpu2goal_rank: Dict[GPUDevice, int], cpu: int, nic: int
    ) -> Tuple[GoalOp, int]:
        if self.consumed:
            raise ValueError(
                "This NCCLPrimitiveSequential has already been consumed and it is single-use."
            )
        self.consumed = True and self.single_use
        ops = [p.to_goal(gpu2goal_rank, cpu, nic) for p in self.primitives]
        max_cpu = max(op[1] for op in ops)
        return GoalSequential(
            list(op[0] for op in ops), gpu2goal_rank[self.gpu], cpu
        ), max_cpu

    def __repr__(self) -> str:
        indent = "  "
        if not self.single_use:
            primitives_repr = [repr(p) for p in self.primitives]
            for i in range(len(primitives_repr)):
                primitives_repr[i] = indent + primitives_repr[i].replace(
                    "\n", "\n" + indent
                )
            return "NCCLPrimitiveSequential(\n" + ",\n".join(primitives_repr) + ")"
        else:
            return f"NCCLPrimitiveSequential(single_use=True)"


class NCCLPrimitive(NCCLPrimitiveComm, ABC):
    """Base class for NCCL primitives."""

    def __init__(
        self,
        context: int,
        gpu: GPUDevice,
        *,
        source_gpu: Optional[GPUDevice] = None,
        target_gpu: Optional[GPUDevice] = None,
        size: int = 0,
        chunk_size: int = 0,
        __proto__: int = -1,
    ):
        super().__init__(gpu)
        self.context = context
        self.source_gpu = source_gpu
        self.target_gpu = target_gpu
        self.size = size
        self.chunk_size = chunk_size if chunk_size > 0 else size
        self.__proto__ = __proto__
        self.additional_info = {}

    def proto_ll(self) -> NCCLPrimitiveComm:
        """Convert to Low-Latency protocol primitives."""
        if self.__proto__ != -1:
            raise ValueError("Reduced operations cannot be converted to LL protocol.")
        n_packets = ceil(self.size / 8)
        return self.__class__(
            self.context,
            self.gpu,
            source_gpu=self.source_gpu,
            target_gpu=self.target_gpu,
            size=n_packets * 8,
            __proto__=0,
            **self.additional_info,
        )

    def proto_simple(self) -> NCCLPrimitiveComm:
        """Convert to Simple protocol primitives."""
        if self.__proto__ != -1:
            raise ValueError(
                "Reduced operations cannot be converted to Simple protocol."
            )
        n_chunks = self.size // self.chunk_size
        if n_chunks <= 1:
            return self.__class__(
                self.context,
                self.gpu,
                source_gpu=self.source_gpu,
                target_gpu=self.target_gpu,
                size=self.size,
                __proto__=2,
                **self.additional_info,
            )

        def generator():
            nonlocal self, n_chunks
            for _ in range(n_chunks):
                yield self.__class__(
                    self.context,
                    self.gpu,
                    source_gpu=self.source_gpu,
                    target_gpu=self.target_gpu,
                    size=self.chunk_size,
                    __proto__=2,
                    **self.additional_info,
                )
            remaining_size = self.size % self.chunk_size
            if remaining_size > 0:
                yield self.__class__(
                    self.context,
                    self.gpu,
                    source_gpu=self.source_gpu,
                    target_gpu=self.target_gpu,
                    size=remaining_size,
                    __proto__=2,
                    **self.additional_info,
                )

        result = NCCLPrimitiveParallel(self.gpu, True, generator())
        return result
        # result = NCCLPrimitiveParallel(self.gpu, single_executer=True)
        # for _ in range(n_chunks):
        #     result.add(self.__class__(
        #         self.context,
        #         self.gpu,
        #         source_gpu=self.source_gpu,
        #         target_gpu=self.target_gpu,
        #         size=self.chunk_size,
        #         __proto__=2
        #     ))

        # remaining_size = self.size % self.chunk_size
        # if remaining_size > 0:
        #     result.add(self.__class__(
        #         self.context,
        #         self.gpu,
        #         source_gpu=self.source_gpu,
        #         target_gpu=self.target_gpu,
        #          size=remaining_size,
        #         __proto__=2
        #     ))
        # return result

    def send_goal(
        self,
        self_goal_rank,
        target_goal_rank: int,
        size: int,
        cpu: int,
        nic: int,
        intra_node: bool,
    ) -> GoalOp:
        if zero_price_communication:
            return GoalSend(target_goal_rank, 0, nic, self.context, self_goal_rank, cpu)
        if intra_node and enable_intra_node_transfer:
            return GoalSequential(
                [
                    GoalCalc(intra_node_transfer_time(size), self_goal_rank, cpu),
                    GoalSend(
                        target_goal_rank, 0, nic, self.context, self_goal_rank, cpu
                    ),
                ],
                self_goal_rank,
                cpu,
            )
        return GoalSend(target_goal_rank, size, nic, self.context, self_goal_rank, cpu)

    def recv_goal(
        self,
        self_goal_rank,
        source_goal_rank: int,
        size: int,
        cpu: int,
        nic: int,
        intra_node: bool,
    ) -> GoalOp:
        if zero_price_communication:
            return GoalRecv(source_goal_rank, 0, nic, self.context, self_goal_rank, cpu)
        if intra_node and enable_intra_node_transfer:
            return GoalSequential(
                [
                    GoalRecv(
                        source_goal_rank, 0, nic, self.context, self_goal_rank, cpu
                    ),
                    GoalCalc(intra_node_transfer_time(size), self_goal_rank, cpu),
                ],
                self_goal_rank,
                cpu,
            )
        return GoalRecv(source_goal_rank, size, nic, self.context, self_goal_rank, cpu)

    @abstractmethod
    def _p_to_goal(
        self,
        gpu2goal_rank: Dict[GPUDevice, int],
        cpu: int,
        nic: int,
        intra_node_send: bool,
        intra_node_recv: bool,
    ) -> GoalOp:
        pass

    def _to_goal(
        self, gpu2goal_rank: Dict[GPUDevice, int], cpu: int, nic: int
    ) -> Tuple[GoalOp, int]:
        # check whether the send and recv are intra-node
        if self.__proto__ == -1:
            raise ValueError("Primitive protocol not specified.")
        intra_send = (
            self.gpu.node_id == self.target_gpu.node_id if self.target_gpu else False
        )
        intra_recv = (
            self.gpu.node_id == self.source_gpu.node_id if self.source_gpu else False
        )
        return self._p_to_goal(gpu2goal_rank, cpu, nic, intra_send, intra_recv), cpu + 1


class NCCLSend(NCCLPrimitive):
    def __repr__(self) -> str:
        return f"NCCLSend(target_gpu={self.target_gpu}, size={self.size})"

    def _p_to_goal(
        self,
        gpu2goal_rank: Dict[GPUDevice, int],
        cpu: int,
        nic: int,
        intra_node_send: bool,
        intra_node_recv: bool,
    ) -> GoalOp:
        return self.send_goal(
            gpu2goal_rank[self.gpu],
            gpu2goal_rank[self.target_gpu],
            self.size,
            cpu,
            nic,
            intra_node_send,
        )


class NCCLRingOp(NCCLPrimitive):
    def __init__(
        self,
        context: int,
        gpu: GPUDevice,
        *,
        source_gpu: Optional[GPUDevice] = None,
        target_gpu: Optional[GPUDevice] = None,
        size: int = 0,
        chunk_size: int = 0,
        __proto__: int = -1,
        reduce_received: bool = False,
        copy_received: bool = False,
    ):
        super().__init__(
            context,
            gpu,
            source_gpu=source_gpu,
            target_gpu=target_gpu,
            size=size,
            chunk_size=chunk_size,
            __proto__=__proto__,
        )
        self.additional_info["reduce_received"] = reduce_received
        self.additional_info["copy_received"] = copy_received

    def __repr__(self) -> str:
        return f"NCCLRingOp(source_gpu={self.source_gpu}, target_gpu={self.target_gpu}, size={self.size})"

    def _p_to_goal(
        self,
        gpu2goal_rank: Dict[GPUDevice, int],
        cpu: int,
        nic: int,
        intra_node_send: bool,
        intra_node_recv: bool,
    ) -> GoalOp:
        self_goal_rank = gpu2goal_rank[self.gpu]
        send = self.send_goal(
            self_goal_rank,
            gpu2goal_rank[self.target_gpu],
            self.size,
            cpu,
            nic,
            intra_node_send,
        )
        recv = self.recv_goal(
            self_goal_rank,
            gpu2goal_rank[self.source_gpu],
            self.size,
            cpu,
            nic,
            intra_node_recv,
        )
        copy_received = self.additional_info.get("copy_received", False)
        reduce_received = self.additional_info.get("reduce_received", False)
        if copy_received or reduce_received:
            calc_time = 0
            if reduce_received:
                calc_time += reduction_time(self.size, self.__proto__)
            if copy_received:
                calc_time += copy_time(self.size, self.__proto__)
            recv = GoalSequential(
                [recv, GoalCalc(calc_time, self_goal_rank, cpu)], self_goal_rank, cpu
            )
        return GoalSequential([send, recv], self_goal_rank, cpu, dependency=False)


class NCCLCopySend(NCCLPrimitive):
    def __repr__(self) -> str:
        return f"NCCLCopySend(target_gpu={self.target_gpu}, size={self.size})"

    def _p_to_goal(
        self,
        gpu2goal_rank: Dict[GPUDevice, int],
        cpu: int,
        nic: int,
        intra_node_send: bool,
        intra_node_recv: bool,
    ) -> GoalOp:
        self_goal_rank = gpu2goal_rank[self.gpu]
        send = self.send_goal(
            self_goal_rank,
            gpu2goal_rank[self.target_gpu],
            self.size,
            cpu,
            nic,
            intra_node_send,
        )
        return GoalSequential(
            [GoalCalc(copy_time(self.size, self.__proto__), self_goal_rank, cpu), send],
            self_goal_rank,
            cpu,
        )


class NCCLRecv(NCCLPrimitive):
    def __repr__(self) -> str:
        return f"NCCLRecv(source_gpu={self.source_gpu}, size={self.size})"

    def _p_to_goal(
        self,
        gpu2goal_rank: Dict[GPUDevice, int],
        cpu: int,
        nic: int,
        intra_node_send: bool,
        intra_node_recv: bool,
    ) -> GoalOp:
        return self.recv_goal(
            gpu2goal_rank[self.gpu],
            gpu2goal_rank[self.source_gpu],
            self.size,
            cpu,
            nic,
            intra_node_recv,
        )


class NCCLSendRecv(NCCLPrimitive):  # for use in ring algorithms
    def __repr__(self) -> str:
        return f"NCCLSendRecv(source_gpu={self.source_gpu}, target_gpu={self.target_gpu}, size={self.size})"

    def _p_to_goal(
        self,
        gpu2goal_rank: Dict[GPUDevice, int],
        cpu: int,
        nic: int,
        intra_node_send: bool,
        intra_node_recv: bool,
    ) -> GoalOp:
        self_goal_rank = gpu2goal_rank[self.gpu]
        send = self.send_goal(
            self_goal_rank,
            gpu2goal_rank[self.target_gpu],
            self.size,
            cpu,
            nic,
            intra_node_send,
        )
        recv = self.recv_goal(
            self_goal_rank,
            gpu2goal_rank[self.source_gpu],
            self.size,
            cpu,
            nic,
            intra_node_recv,
        )
        return GoalParallel([send, recv], self_goal_rank, cpu)


class NCCLReduce(NCCLPrimitive):
    def __repr__(self) -> str:
        return f"NCCLReduce(source_gpu={self.source_gpu}, size={self.size})"

    def _p_to_goal(
        self,
        gpu2goal_rank: Dict[GPUDevice, int],
        cpu: int,
        nic: int,
        intra_node_send: bool,
        intra_node_recv: bool,
    ) -> GoalOp:
        self_goal_rank = gpu2goal_rank[self.gpu]
        return GoalCalc(reduction_time(self.size, self.__proto__), self_goal_rank, cpu)


class NCCLCopy(NCCLPrimitive):
    def __repr__(self) -> str:
        return f"NCCLCopy(size={self.size})"

    def _p_to_goal(
        self,
        gpu2goal_rank: Dict[GPUDevice, int],
        cpu: int,
        nic: int,
        intra_node_send: bool,
        intra_node_recv: bool,
    ) -> GoalOp:
        self_goal_rank = gpu2goal_rank[self.gpu]
        return GoalCalc(copy_time(self.size, self.__proto__), self_goal_rank, cpu)


class NCCLReduceCopy(NCCLPrimitive):
    def __repr__(self) -> str:
        return f"NCCLReduceCopy(size={self.size})"

    def _p_to_goal(
        self,
        gpu2goal_rank: Dict[GPUDevice, int],
        cpu: int,
        nic: int,
        intra_node_send: bool,
        intra_node_recv: bool,
    ) -> GoalOp:
        self_goal_rank = gpu2goal_rank[self.gpu]
        return GoalCalc(
            reduction_time(self.size, self.__proto__)
            + copy_time(self.size, self.__proto__),
            self_goal_rank,
            cpu,
        )


class NCCLRecvReduce(NCCLPrimitive):
    def __repr__(self) -> str:
        return f"NCCLRecvReduce(source_gpu={self.source_gpu}, size={self.size})"

    def _p_to_goal(
        self,
        gpu2goal_rank: Dict[GPUDevice, int],
        cpu: int,
        nic: int,
        intra_node_send: bool,
        intra_node_recv: bool,
    ) -> GoalOp:
        self_goal_rank = gpu2goal_rank[self.gpu]
        recv = self.recv_goal(
            self_goal_rank,
            gpu2goal_rank[self.source_gpu],
            self.size,
            cpu,
            nic,
            intra_node_recv,
        )
        return GoalSequential(
            [
                recv,
                GoalCalc(
                    reduction_time(self.size, self.__proto__), self_goal_rank, cpu
                ),
            ],
            self_goal_rank,
            cpu,
        )


class NCCLRecvReduceCopy(NCCLPrimitive):
    def __repr__(self) -> str:
        return f"NCCLRecvReduceCopy(source_gpu={self.source_gpu}, size={self.size})"

    def _p_to_goal(
        self,
        gpu2goal_rank: Dict[GPUDevice, int],
        cpu: int,
        nic: int,
        intra_node_send: bool,
        intra_node_recv: bool,
    ) -> GoalOp:
        self_goal_rank = gpu2goal_rank[self.gpu]
        recv = self.recv_goal(
            self_goal_rank,
            gpu2goal_rank[self.source_gpu],
            self.size,
            cpu,
            nic,
            intra_node_recv,
        )
        return GoalSequential(
            [
                recv,
                GoalCalc(
                    reduction_time(self.size, self.__proto__)
                    + copy_time(self.size, self.__proto__),
                    self_goal_rank,
                    cpu,
                ),
            ],
            self_goal_rank,
            cpu,
        )


class NCCLRecvCopySend(NCCLPrimitive):
    def __repr__(self) -> str:
        return f"NCCLRecvCopySend(source_gpu={self.source_gpu}, target_gpu={self.target_gpu}, size={self.size})"

    def _p_to_goal(
        self,
        gpu2goal_rank: Dict[GPUDevice, int],
        cpu: int,
        nic: int,
        intra_node_send: bool,
        intra_node_recv: bool,
    ) -> GoalOp:
        self_goal_rank = gpu2goal_rank[self.gpu]
        recv = self.recv_goal(
            self_goal_rank,
            gpu2goal_rank[self.source_gpu],
            self.size,
            cpu,
            nic,
            intra_node_recv,
        )
        send = self.send_goal(
            self_goal_rank,
            gpu2goal_rank[self.target_gpu],
            self.size,
            cpu,
            nic,
            intra_node_send,
        )
        return GoalSequential(
            [
                recv,
                GoalCalc(copy_time(self.size, self.__proto__), self_goal_rank, cpu),
                send,
            ],
            self_goal_rank,
            cpu,
        )


class NCCLRecvCopy(NCCLPrimitive):
    def __repr__(self):
        return f"NCCLRecvCopy(source_gpu={self.source_gpu}, size={self.size})"

    def _p_to_goal(
        self,
        gpu2goal_rank: Dict[GPUDevice, int],
        cpu: int,
        nic: int,
        intra_node_send: bool,
        intra_node_recv: bool,
    ) -> GoalOp:
        self_goal_rank = gpu2goal_rank[self.gpu]
        recv = self.recv_goal(
            self_goal_rank,
            gpu2goal_rank[self.source_gpu],
            self.size,
            cpu,
            nic,
            intra_node_recv,
        )
        return GoalSequential(
            [recv, GoalCalc(copy_time(self.size, self.__proto__), self_goal_rank, cpu)],
            self_goal_rank,
            cpu,
        )


class NCCLRecvReduceSend(NCCLPrimitive):
    def __repr__(self):
        return f"NCCLRecvReduceSend(source_gpu={self.source_gpu.id}, target_gpu={self.target_gpu.id}, size={self.size})"

    def _p_to_goal(
        self,
        gpu2goal_rank: Dict[GPUDevice, int],
        cpu: int,
        nic: int,
        intra_node_send: bool,
        intra_node_recv: bool,
    ) -> GoalOp:
        self_goal_rank = gpu2goal_rank[self.gpu]
        recv = self.recv_goal(
            self_goal_rank,
            gpu2goal_rank[self.source_gpu],
            self.size,
            cpu,
            nic,
            intra_node_recv,
        )
        send = self.send_goal(
            self_goal_rank,
            gpu2goal_rank[self.target_gpu],
            self.size,
            cpu,
            nic,
            intra_node_send,
        )
        return GoalSequential(
            [
                recv,
                GoalCalc(
                    reduction_time(self.size, self.__proto__), self_goal_rank, cpu
                ),
                send,
            ],
            self_goal_rank,
            cpu,
        )


class NCCLRecvReduceCopySend(NCCLPrimitive):
    def __repr__(self):
        return f"NCCLRecvReduceCopySend(source_gpu={self.source_gpu.id}, target_gpu={self.target_gpu.id}, size={self.size})"

    def _p_to_goal(
        self,
        gpu2goal_rank: Dict[GPUDevice, int],
        cpu: int,
        nic: int,
        intra_node_send: bool,
        intra_node_recv: bool,
    ) -> GoalOp:
        self_goal_rank = gpu2goal_rank[self.gpu]
        recv = self.recv_goal(
            self_goal_rank,
            gpu2goal_rank[self.source_gpu],
            self.size,
            cpu,
            nic,
            intra_node_recv,
        )
        send = self.send_goal(
            self_goal_rank,
            gpu2goal_rank[self.target_gpu],
            self.size,
            cpu,
            nic,
            intra_node_send,
        )
        return GoalSequential(
            [
                recv,
                GoalCalc(
                    reduction_time(self.size, self.__proto__)
                    + copy_time(self.size, self.__proto__),
                    self_goal_rank,
                    cpu,
                ),
                send,
            ],
            self_goal_rank,
            cpu,
        )
