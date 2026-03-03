from dataclasses import dataclass
from typing import List, Tuple, Union, Optional, Dict
from enum import Enum
from collections.abc import Hashable
from abc import ABC, abstractmethod
from contextlib import ContextDecorator
import contextvars
from goal import GoalOp, GoalParallel, GoalSequential, GoalSend, GoalRecv, GoalCalc

RankId = Hashable  # e.g., "cpu", ("gpu", 0), ("nic", 0)

class CollAlgo(Enum):
    TREE = 0
    RING = 1
    RECURSIVE_DOUBLING = 2

class Communicator:
    def __init__(self, comm_id: str, device_ids: List[RankId]):
        self.comm_id: str = comm_id
        self.rank2device_id: List[RankId] = device_ids
        self.device_id2rank: Dict[RankId, int] = {d: r for r, d in enumerate(device_ids)}
        self.size: int = len(device_ids)
        self.rank2device_id.append(None) # for invalid rank
    
    def __repr__(self):
        return f"Communicator(id={self.comm_id}, ranks={self.rank2device_id[:-1]})"


DEFAULT_COMM_DEVICE_ID: contextvars.ContextVar[Optional[RankId]] = contextvars.ContextVar("default_comm_device", default=None)
class CollDevice(ContextDecorator):
    """Context manager for collective communication device context.

    Usage:
        with CollDevice(comm, rank):
            # Inside this block, the current collective device context is set to comm and rank.
            # This can be used by CommOp instances to determine their device context.
            ...
    """
    def __init__(self, device_id):
        self.device_id = device_id
    
    def __enter__(self):
        self._token_self = DEFAULT_COMM_DEVICE_ID.set(self.device_id)
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self._token_self is not None:
            DEFAULT_COMM_DEVICE_ID.reset(self._token_self)
        return False  # Don't suppress exceptions

def _get_current_device() -> Optional[RankId]:
    """Get the current collective communication device context, if any."""
    return DEFAULT_COMM_DEVICE_ID.get()


class CommOp(ABC):
    def __init__(self, comm: Communicator, context: int, device_id: Optional[RankId] = None):
        self.comm = comm
        self.context = context
        if device_id is None:
            # If no explicit device_id is provided, use the current collective device context
            device_id = _get_current_device()
        self.device_id = device_id  # for scheduling; can be None for collective ops
    
    @abstractmethod
    def _to_goal(self, device_id2goal_rank: Dict[RankId, int], curr_device: RankId, starting_cpu_id: int, nic: int) -> GoalOp:
        """Translate this CommOp into a GoalOp for scheduling."""
        pass

    def to_goal(self, device_id2goal_rank: Dict[RankId, int], starting_cpu_id: int, nic: int) -> GoalOp:
        device_id = self.device_id if self.device_id is not None else _get_current_device()
        if device_id is None:
            raise ValueError("CommOp requires a device_id, but none was provided and no collective device context is set.")
        return self._to_goal(device_id2goal_rank, device_id, starting_cpu_id, nic)

    def __repr__(self):
        return f"CommOp(comm={self.comm}, context={self.context}, device_id={self.device_id})"

class P2POp(CommOp):
    def __init__(self, comm: Communicator, size: int, context: int, peer_rank: int, device_id: Optional[RankId] = None):
        super().__init__(comm, context, device_id)
        self.peer_rank = peer_rank
        self.size = size

    def _to_goal(self, device_id2goal_rank, curr_device, starting_cpu_id, nic) -> GoalOp:
        peer_goal_rank = device_id2goal_rank[self.comm.rank2device_id[self.peer_rank]]
        self_goal_rank = device_id2goal_rank[curr_device]
        return self._goal_cls(peer_goal_rank, self.size, nic, self.context, self_goal_rank, starting_cpu_id)

class Send(P2POp):
    _goal_cls = GoalSend

class Recv(P2POp):
    _goal_cls = GoalRecv

class CollOp(CommOp):
    def __init__(self, comm: Communicator, size: int, context: int, algo: CollAlgo = CollAlgo.RING, device_id: Optional[RankId] = None):
        super().__init__(comm, context, device_id)
        self.algo = algo
        self.size = size

class ReduceScatter(CollOp):
    def _to_goal(self, device_id2goal_rank, curr_device, starting_cpu_id, nic) -> GoalOp:
        curr_rank = self.comm.device_id2rank.get(curr_device, -1)
        if curr_rank == -1:
            raise ValueError(f"Current device {curr_device} is not part of the communicator {self.comm}")
        chunk_size = self.size // self.comm.size
        self_goal_rank = device_id2goal_rank[curr_device]
        assert chunk_size * self.comm.size == self.size, "size must be divisible by comm size"
        if self.algo == CollAlgo.RING:
            # ring algorithm
            prev_rank = (curr_rank - 1) % self.comm.size
            prev_id = self.comm.rank2device_id[prev_rank]
            prev_goal_rank = device_id2goal_rank[prev_id]
            next_rank = (curr_rank + 1) % self.comm.size
            next_id = self.comm.rank2device_id[next_rank]
            next_goal_rank = device_id2goal_rank[next_id]
            # For ring algorithm, there will be comm.size - 1 steps of send/recv
            result = []
            for step in range(self.comm.size - 1):
                result.append(GoalParallel([
                    GoalRecv(prev_goal_rank, chunk_size, nic, self.context, self_goal_rank, starting_cpu_id),
                    GoalSend(next_goal_rank, chunk_size, nic, self.context, self_goal_rank, starting_cpu_id)
                ]))
            return GoalSequential(result, self_goal_rank, starting_cpu_id)
        elif self.algo == CollAlgo.RECURSIVE_DOUBLING:
            # recursive halving algorithm for reduce-scatter:
            assert self.comm.size & (self.comm.size - 1) == 0, "comm size must be a power of 2 for recursive doubling"
            result = []
            mask = self.comm.size >> 1  # start with most distant partner
            curr_chunk_size = self.size
            while mask >= 1:
                partner_rank = curr_rank ^ mask
                if partner_rank < self.comm.size:
                    partner_id = self.comm.rank2device_id[partner_rank]
                    partner_goal_rank = device_id2goal_rank[partner_id]
                    result.append(GoalParallel([
                        GoalRecv(partner_goal_rank, curr_chunk_size // 2, nic, self.context, self_goal_rank, starting_cpu_id),
                        GoalSend(partner_goal_rank, curr_chunk_size // 2, nic, self.context, self_goal_rank, starting_cpu_id)
                    ]))
                curr_chunk_size //= 2
                mask >>= 1
            return GoalSequential(result, self_goal_rank, starting_cpu_id)
        else:
            raise NotImplementedError(f"CollAlgo {self.algo} is not implemented yet")

class AllGather(CollOp):
    def _to_goal(self, device_id2goal_rank, curr_device, starting_cpu_id, nic) -> GoalOp:
        curr_rank = self.comm.device_id2rank.get(curr_device, -1)
        if curr_rank == -1:
            raise ValueError(f"Current device {curr_device} is not part of the communicator {self.comm}")
        chunk_size = self.size // self.comm.size
        self_goal_rank = device_id2goal_rank[curr_device]
        assert chunk_size * self.comm.size == self.size, "size must be divisible by comm size"
        if self.algo == CollAlgo.RING:
            # ring algorithm
            prev_rank = (curr_rank - 1) % self.comm.size
            prev_id = self.comm.rank2device_id[prev_rank]
            prev_goal_rank = device_id2goal_rank[prev_id]
            next_rank = (curr_rank + 1) % self.comm.size
            next_id = self.comm.rank2device_id[next_rank]
            next_goal_rank = device_id2goal_rank[next_id]
            # For ring algorithm, there will be comm.size - 1 steps of send/recv
            result = []
            for step in range(self.comm.size - 1):
                result.append(GoalParallel([
                    GoalRecv(prev_goal_rank, chunk_size, nic, self.context, self_goal_rank, starting_cpu_id),
                    GoalSend(next_goal_rank, chunk_size, nic, self.context, self_goal_rank, starting_cpu_id)
                ]))
            return GoalSequential(result, self_goal_rank, starting_cpu_id)
        elif self.algo == CollAlgo.RECURSIVE_DOUBLING:
            # recursive doubling algorithm for all-gather:
            # inverse of recursive halving reduce-scatter — start with nearest
            # partner so the working set doubles each step until all data is held.
            assert self.comm.size & (self.comm.size - 1) == 0, "comm size must be a power of 2 for recursive doubling"
            result = []
            mask = 1  # start with nearest partner
            curr_chunk_size = chunk_size  # each rank starts with 1 chunk
            while mask < self.comm.size:
                partner_rank = curr_rank ^ mask
                if partner_rank < self.comm.size:
                    partner_id = self.comm.rank2device_id[partner_rank]
                    partner_goal_rank = device_id2goal_rank[partner_id]
                    result.append(GoalParallel([
                        GoalRecv(partner_goal_rank, curr_chunk_size, nic, self.context, self_goal_rank, starting_cpu_id),
                        GoalSend(partner_goal_rank, curr_chunk_size, nic, self.context, self_goal_rank, starting_cpu_id)
                    ]))
                curr_chunk_size *= 2
                mask <<= 1
            return GoalSequential(result, self_goal_rank, starting_cpu_id)
        else:
            raise NotImplementedError(f"CollAlgo {self.algo} is not implemented yet")

class AllReduce(CollOp):
    def _to_goal(self, device_id2goal_rank, curr_device, starting_cpu_id, nic) -> GoalOp:
        curr_rank = self.comm.device_id2rank.get(curr_device, -1)
        if curr_rank == -1:
            raise ValueError(f"Current device {curr_device} is not part of the communicator {self.comm}")
        chunk_size = self.size // self.comm.size
        self_goal_rank = device_id2goal_rank[curr_device]
        assert chunk_size * self.comm.size == self.size, "size must be divisible by comm size"
        if self.algo == CollAlgo.RING:
            # ring algorithm: (N-1) reduce-scatter steps then (N-1) all-gather steps
            prev_rank = (curr_rank - 1) % self.comm.size
            prev_goal_rank = device_id2goal_rank[self.comm.rank2device_id[prev_rank]]
            next_rank = (curr_rank + 1) % self.comm.size
            next_goal_rank = device_id2goal_rank[self.comm.rank2device_id[next_rank]]
            result = []
            for step in range(2 * (self.comm.size - 1)):
                result.append(GoalParallel([
                    GoalRecv(prev_goal_rank, chunk_size, nic, self.context, self_goal_rank, starting_cpu_id),
                    GoalSend(next_goal_rank, chunk_size, nic, self.context, self_goal_rank, starting_cpu_id)
                ]))
            return GoalSequential(result, self_goal_rank, starting_cpu_id)
        elif self.algo == CollAlgo.RECURSIVE_DOUBLING:
            # recursive halving reduce-scatter then recursive doubling all-gather
            assert self.comm.size & (self.comm.size - 1) == 0, "comm size must be a power of 2 for recursive doubling"
            result = []
            # phase 1: reduce-scatter (halving, large-to-small, most distant first)
            mask = self.comm.size >> 1
            curr_chunk_size = self.size
            while mask >= 1:
                partner_rank = curr_rank ^ mask
                if partner_rank < self.comm.size:
                    partner_goal_rank = device_id2goal_rank[self.comm.rank2device_id[partner_rank]]
                    result.append(GoalParallel([
                        GoalRecv(partner_goal_rank, curr_chunk_size // 2, nic, self.context, self_goal_rank, starting_cpu_id),
                        GoalSend(partner_goal_rank, curr_chunk_size // 2, nic, self.context, self_goal_rank, starting_cpu_id)
                    ]))
                curr_chunk_size //= 2
                mask >>= 1
            # phase 2: all-gather (doubling, small-to-large, nearest first)
            mask = 1
            curr_chunk_size = chunk_size
            while mask < self.comm.size:
                partner_rank = curr_rank ^ mask
                if partner_rank < self.comm.size:
                    partner_goal_rank = device_id2goal_rank[self.comm.rank2device_id[partner_rank]]
                    result.append(GoalParallel([
                        GoalRecv(partner_goal_rank, curr_chunk_size, nic, self.context, self_goal_rank, starting_cpu_id),
                        GoalSend(partner_goal_rank, curr_chunk_size, nic, self.context, self_goal_rank, starting_cpu_id)
                    ]))
                curr_chunk_size *= 2
                mask <<= 1
            return GoalSequential(result, self_goal_rank, starting_cpu_id)
        else:
            raise NotImplementedError(f"CollAlgo {self.algo} is not implemented yet")
