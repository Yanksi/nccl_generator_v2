from dataclasses import dataclass
from typing import List, Tuple, Union, Optional, Dict
from enum import Enum
from collections.abc import Hashable
from abc import ABC, abstractmethod
from contextlib import ContextDecorator
import contextvars
from goal import GoalOp, GoalParallel, GoalSequential, GoalSend, GoalRecv, GoalCalc, GoalRank, GoalCPU, DEFAULT_SELF_RANK

RankId = Hashable  # e.g., "cpu", ("gpu", 0), ("nic", 0)

class CollAlgo(Enum):
    TREE = 0
    RING = 1
    RECURSIVE_DOUBLING = 2


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
        self._token_goal = DEFAULT_SELF_RANK.set(self.device_id)  # also set GoalRank context for convenience
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self._token_self is not None:
            DEFAULT_COMM_DEVICE_ID.reset(self._token_self)
        if self._token_goal is not None:
            DEFAULT_SELF_RANK.reset(self._token_goal)
        return False  # Don't suppress exceptions
    
    def __repr__(self):
        return f"CollDevice({self.device_id})"
    
    def __hash__(self):
        return hash(self.device_id)
    
    def __eq__(self, other):
        if not isinstance(other, CollDevice):
            return self.device_id == other
        return self.device_id == other.device_id

def _get_current_device() -> Optional[RankId]:
    """Get the current collective communication device context, if any."""
    return DEFAULT_COMM_DEVICE_ID.get()

class Communicator:
    def __init__(self, devices: List[Union[RankId, CollDevice]], comm_id: str = None):
        if comm_id is None:
            comm_id = f"comm_{id(self)}"
        self.comm_id: str = comm_id
        self.rank2device_id: List[RankId] = [device.device_id if isinstance(device, CollDevice) else device for device in devices]
        self.device_id2rank: Dict[RankId, int] = {d: r for r, d in enumerate(self.rank2device_id)}
        self.size: int = len(self.rank2device_id)
        self.rank2device_id.append(None) # for invalid rank
    
    def __repr__(self):
        return f"Communicator(id={self.comm_id}, ranks={self.rank2device_id[:-1]})"
    
    def allgather(self, size: int, context: int, algo: CollAlgo = CollAlgo.RING, device_id: Optional[RankId] = None):
        return AllGather(self, size, context, algo, device_id)
    
    def allreduce(self, size: int, context: int, algo: CollAlgo = CollAlgo.RING, device_id: Optional[RankId] = None):
        return AllReduce(self, size, context, algo, device_id)

    def reducescatter(self, size: int, context: int, algo: CollAlgo = CollAlgo.RING, device_id: Optional[RankId] = None):
        return ReduceScatter(self, size, context, algo, device_id)
    
    def alltoall(self, size: int, context: int, algo: CollAlgo = CollAlgo.RING, device_id: Optional[RankId] = None):
        return AllToAll(self, size, context, algo, device_id)
    
    def send(self, size: int, context: int, dst_rank: int, device_id: Optional[RankId] = None):
        return Send(self, size, context, dst_rank, device_id)
    
    def recv(self, size: int, context: int, src_rank: int, device_id: Optional[RankId] = None):
        return Recv(self, size, context, src_rank, device_id)

class CommOp(ABC):
    def __init__(self, comm: Communicator, context: int, device_id: Optional[Union[RankId, CollDevice]] = None):
        self.comm = comm
        self.context = context
        if isinstance(device_id, CollDevice):
            device_id = device_id.device_id
        if device_id is None:
            # If no explicit device_id is provided, use the current collective device context
            device_id = _get_current_device()
        self.device_id = device_id  # for scheduling; can be None for collective ops
    
    @abstractmethod
    def _to_goal(self, device_id2goal_rank: Dict[RankId, int], curr_device: RankId, starting_cpu_id: int, nic: int) -> GoalOp:
        """Translate this CommOp into a GoalOp for scheduling."""
        pass

    def to_goal(self, device2goal_rank: Dict[Union[RankId, CollDevice], Union[int, GoalRank]], starting_cpu_id: int, nic: int) -> GoalOp:
        device_id = self.device_id if self.device_id is not None else _get_current_device()
        if device_id is None:
            raise ValueError("CommOp requires a device_id, but none was provided and no collective device context is set.")
        device_id2goal_rank = {d.device_id if isinstance(d, CollDevice) else d: r.self_rank if isinstance(r, GoalRank) else r for d, r in device2goal_rank.items()}
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

    def _setup(self, device_id2goal_rank, curr_device):
        curr_rank = self.comm.device_id2rank.get(curr_device, -1)
        if curr_rank == -1:
            raise ValueError(f"Current device {curr_device} is not part of the communicator {self.comm}")
        chunk_size = self.size // self.comm.size
        assert chunk_size * self.comm.size == self.size, "size must be divisible by comm size"
        self_goal_rank = device_id2goal_rank[curr_device]
        return curr_rank, chunk_size, self_goal_rank

    def _ring_steps(self, n_steps, curr_rank, chunk_size, self_goal_rank, device_id2goal_rank, starting_cpu_id, nic):
        """Ring send/recv steps (identical pattern for both ReduceScatter and AllGather)."""
        prev_goal_rank = device_id2goal_rank[self.comm.rank2device_id[(curr_rank - 1) % self.comm.size]]
        next_goal_rank = device_id2goal_rank[self.comm.rank2device_id[(curr_rank + 1) % self.comm.size]]
        return [GoalParallel([
            GoalRecv(prev_goal_rank, chunk_size, nic, self.context, self_goal_rank, starting_cpu_id),
            GoalSend(next_goal_rank, chunk_size, nic, self.context, self_goal_rank, starting_cpu_id)
        ], self_goal_rank, starting_cpu_id) for _ in range(n_steps)]

    def _recursive_steps(self, curr_rank, chunk_size, self_goal_rank, device_id2goal_rank, starting_cpu_id, nic, *, halving: bool):
        """Unified recursive halving/doubling steps.
        halving=True:  ReduceScatter pattern — most distant partner first, chunk shrinks each step.
        halving=False: AllGather pattern    — nearest partner first,  chunk grows  each step.
        """
        assert self.comm.size & (self.comm.size - 1) == 0, "comm size must be a power of 2 for recursive doubling"
        masks = [1 << i for i in range(self.comm.size.bit_length() - 1)]  # [1, 2, ..., N/2]
        if halving:
            masks = reversed(masks)
        curr_chunk_size = self.size // 2 if halving else chunk_size
        result = []
        for mask in masks:
            partner_rank = curr_rank ^ mask
            if partner_rank < self.comm.size:
                partner_goal_rank = device_id2goal_rank[self.comm.rank2device_id[partner_rank]]
                result.append(GoalParallel([
                    GoalRecv(partner_goal_rank, curr_chunk_size, nic, self.context, self_goal_rank, starting_cpu_id),
                    GoalSend(partner_goal_rank, curr_chunk_size, nic, self.context, self_goal_rank, starting_cpu_id)
                ], self_goal_rank, starting_cpu_id))
            curr_chunk_size = curr_chunk_size // 2 if halving else curr_chunk_size * 2
        return result


class ReduceScatter(CollOp):
    def _to_goal(self, device_id2goal_rank, curr_device, starting_cpu_id, nic) -> GoalOp:
        curr_rank, chunk_size, self_goal_rank = self._setup(device_id2goal_rank, curr_device)
        if self.algo == CollAlgo.RING:
            steps = self._ring_steps(self.comm.size - 1, curr_rank, chunk_size, self_goal_rank, device_id2goal_rank, starting_cpu_id, nic)
        elif self.algo == CollAlgo.RECURSIVE_DOUBLING:
            steps = self._recursive_steps(curr_rank, chunk_size, self_goal_rank, device_id2goal_rank, starting_cpu_id, nic, halving=True)
        else:
            raise NotImplementedError(f"CollAlgo {self.algo} is not implemented yet")
        return GoalSequential(steps, self_goal_rank, starting_cpu_id)

class AllGather(CollOp):
    def _to_goal(self, device_id2goal_rank, curr_device, starting_cpu_id, nic) -> GoalOp:
        curr_rank, chunk_size, self_goal_rank = self._setup(device_id2goal_rank, curr_device)
        if self.algo == CollAlgo.RING:
            steps = self._ring_steps(self.comm.size - 1, curr_rank, chunk_size, self_goal_rank, device_id2goal_rank, starting_cpu_id, nic)
        elif self.algo == CollAlgo.RECURSIVE_DOUBLING:
            steps = self._recursive_steps(curr_rank, chunk_size, self_goal_rank, device_id2goal_rank, starting_cpu_id, nic, halving=False)
        else:
            raise NotImplementedError(f"CollAlgo {self.algo} is not implemented yet")
        return GoalSequential(steps, self_goal_rank, starting_cpu_id)

class AllReduce(CollOp):
    def _to_goal(self, device_id2goal_rank, curr_device, starting_cpu_id, nic) -> GoalOp:
        curr_rank, chunk_size, self_goal_rank = self._setup(device_id2goal_rank, curr_device)
        if self.algo == CollAlgo.RING:
            steps = self._ring_steps(2 * (self.comm.size - 1), curr_rank, chunk_size, self_goal_rank, device_id2goal_rank, starting_cpu_id, nic)
        elif self.algo == CollAlgo.RECURSIVE_DOUBLING:
            steps = (self._recursive_steps(curr_rank, chunk_size, self_goal_rank, device_id2goal_rank, starting_cpu_id, nic, halving=True) +
                     self._recursive_steps(curr_rank, chunk_size, self_goal_rank, device_id2goal_rank, starting_cpu_id, nic, halving=False))
        else:
            raise NotImplementedError(f"CollAlgo {self.algo} is not implemented yet")
        return GoalSequential(steps, self_goal_rank, starting_cpu_id)

class AllToAll(CollOp):
    def _to_goal(self, device_id2goal_rank, curr_device, starting_cpu_id, nic) -> GoalOp:
        # a simple linear send/recv pattern; can be optimized with better algorithms if needed
        curr_rank, chunk_size, self_goal_rank = self._setup(device_id2goal_rank, curr_device)
        comms = []
        for peer_rank in range(self.comm.size):
            if peer_rank == curr_rank:
                continue
            peer_goal_rank = device_id2goal_rank[self.comm.rank2device_id[peer_rank]]
            comms.extend(
                [GoalRecv(peer_goal_rank, chunk_size, nic, self.context, self_goal_rank, starting_cpu_id),
                GoalSend(peer_goal_rank, chunk_size, nic, self.context, self_goal_rank, starting_cpu_id)]
            )
        return GoalParallel(comms, self_goal_rank, starting_cpu_id)

if __name__ == "__main__":
    # some test code and examples of usage
    devices = [CollDevice(("gpu", i)) for i in range(4)]
    comm = Communicator(devices, "comm1")
    rs = ReduceScatter(comm, size=1024, context=0, algo=CollAlgo.RECURSIVE_DOUBLING)
    test_file = "test_simple.goal"
    device2goal_rank = {d: i for i, d in enumerate(devices)}
    with open(test_file, "w") as f:
        for device, rank in device2goal_rank.items():
            with device:
                goal_op = rs.to_goal(device2goal_rank, starting_cpu_id=0, nic=0)
            f.write(f"rank {rank}: {{\n")
            f.writelines(f"  {line}\n" for line in goal_op.generate_lines())
            f.write("}\n")