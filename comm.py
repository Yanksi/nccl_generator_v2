from dataclasses import dataclass
from typing import List, Tuple, Union, Optional, Dict
from enum import Enum
from collections.abc import Hashable
from abc import ABC, abstractmethod
from contextlib import ContextDecorator
import contextvars
from goal import GoalOp

RankId = Hashable  # e.g., "cpu", ("gpu", 0), ("nic", 0)

class CollAlgo(Enum):
    TREE = 0
    RING = 1

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
    def _to_goal(self, device_id2goal_rank: Dict[RankId, int], starting_cpu_id: int, nic: int) -> GoalOp:
        """Translate this CommOp into a GoalOp for scheduling."""
        pass

    def to_goal(self, device_id2goal_rank: Dict[RankId, int], starting_cpu_id: int, nic: int) -> GoalOp:
        if self.device_id is None:
            self.device_id = _get_current_device()
        if self.device_id is None:
            raise ValueError("CommOp requires a device_id, but none was provided and no collective device context is set.")
        return self._to_goal(device_id2goal_rank, starting_cpu_id, nic)

    def __repr__(self):
        return f"CommOp(comm={self.comm}, context={self.context}, device_id={self.device_id})"