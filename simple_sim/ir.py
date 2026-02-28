from __future__ import annotations

import enum
from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Tuple, TypedDict, Union
import itertools

_id_counter = itertools.count()

# Stack of active checkpoint region IDs.  Pushed / popped by the
# ``checkpoint()`` context manager in ops_compute.  Used by
# ``Tensor.__post_init__`` to automatically tag tensors created
# inside a checkpoint region.
_active_checkpoint_stack: list[int] = []


class CostMeta(TypedDict):
    """Cost metadata for roofline model analysis."""
    flops: int      # floating point operations
    mem_read: int   # memory read in elements
    mem_write: int  # memory write in elements


class MemoryCategory(enum.Enum):
    """How a tensor's memory is treated during execution."""
    MATERIALIZED = "materialized"          # actually allocated & stored
    NOT_MATERIALIZED = "not_materialized"  # zero-cost view / alias (reshape, ones_like)
    RECOMPUTED = "recomputed"              # inside checkpoint region, discarded then recomputed


# -------- parallel metadata --------

@dataclass(frozen=True)
class Group:
    kind: str  # "dp" | "tp" | "pp" | etc.
    id: int
    size: int

@dataclass(frozen=True)
class ShardSpec:
    kind: str  # "replicated" | "sharded"
    axis: Optional[int] = None
    parts: Optional[int] = None


# -------- IR values --------

@dataclass(frozen=True, eq=False)
class Tensor:
    """
    Immutable tensor value in the computation graph.
    
    Uses identity-based equality (eq=False) to avoid infinite recursion
    when hashing tensors that are part of cyclic graphs (e.g., multi-iteration
    training where iteration N's params depend on iteration N-1's outputs).
    """
    shape: Tuple[int, ...]
    dtype: str = "fp16"
    producer: Optional["OpNode"] = None
    requires_grad: bool = False
    name: Optional[str] = None

    shard: ShardSpec = field(default_factory=lambda: ShardSpec("replicated"))
    tp_group: Optional[Group] = None
    dp_group: Optional[Group] = None
    # Set automatically when the tensor is created inside a ``checkpoint()``
    # region.  ``None`` means the activation is stored during the forward pass;
    # an integer means it belongs to that checkpoint region and will be
    # discarded (then recomputed during backward).
    checkpoint_region_id: Optional[int] = None

    # Memory category: whether the tensor is materialized, a zero-cost view,
    # or will be recomputed.  Set by the producing op or by __post_init__.
    memory_category: MemoryCategory = MemoryCategory.MATERIALIZED

    # True for tensors produced during the backward pass (gradient tensors).
    # Used for visualization only – memory lifecycle is the same as forward.
    is_gradient: bool = False

    # For NOT_MATERIALIZED tensors: the tensor whose memory this one shares.
    # A reshape aliases its input; detach / wait_for alias their input; etc.
    # ``None`` means the tensor owns its own memory (MATERIALIZED / RECOMPUTED)
    # or is a costless constant (ones_like).
    aliases: Optional["Tensor"] = None

    def __post_init__(self) -> None:
        # Auto-tag tensors created inside a checkpoint region.
        if self.checkpoint_region_id is None and _active_checkpoint_stack:
            object.__setattr__(
                self, "checkpoint_region_id", _active_checkpoint_stack[-1],
            )

    def resolve_alias(self) -> "Tensor":
        """Follow the alias chain to the memory-owning (root) tensor."""
        t = self
        while t.aliases is not None:
            t = t.aliases
        return t

    def physical_shape(self) -> Tuple[int, ...]:
        """Return the actual per-rank shape, accounting for sharding.

        If the tensor is replicated (or has no shard spec), the physical
        shape equals the logical shape.  If it is sharded along some axis,
        that axis is divided by ``shard.parts``.
        """
        if self.shard.kind == "replicated" or self.shard.axis is None or self.shard.parts is None:
            return self.shape
        s = list(self.shape)
        axis = self.shard.axis % len(s)  # normalize negative axes
        s[axis] //= self.shard.parts
        return tuple(s)

@dataclass(frozen=True, eq=False)
class Token:
    """
    Token value representing a control dependency.
    Uses identity-based equality (eq=False) for the same reason as Tensor.
    """
    producer: "OpNode"
    name: Optional[str] = None

Value = Union[Tensor, Token]


# -------- Ops and nodes --------

@dataclass(frozen=True, eq=False)
class OpNode:
    """
    Base class for all operations in the computation graph.
    
    Each op type is a subclass of OpNode with operation-specific parameters.
    Uses identity-based equality (eq=False) to avoid infinite recursion
    when hashing nodes in cyclic graphs.
    
    Subclasses should override:
    - kind: str class attribute for the operation type
    - vjp(): for gradient computation
    - get_cost_meta(): for cost modeling (returns CostMeta dict)
    """
    inputs: Tuple[Value, ...] = ()
    outputs: Tuple[Value, ...] = ()
    id: int = field(default_factory=lambda: next(_id_counter))
    
    # Override in subclasses
    kind: str = field(default="op", init=False)
    
    def vjp(self, grad_output: "Tensor") -> Dict["Tensor", "Tensor"]:
        """
        Compute vector-Jacobian product for this op.
        Returns a dict mapping input tensors to their gradients.
        Override in subclasses that support differentiation.
        """
        return {}

    def get_cost_meta(self) -> CostMeta:
        """
        Return cost metadata for roofline model analysis.
        Override in subclasses. Returns dict with flops, mem_read, mem_write.
        """
        return {"flops": 0, "mem_read": 0, "mem_write": 0}


@dataclass(frozen=True, eq=False)
class ComputeOp(OpNode):
    """Base class for compute operations (runs on GPU compute units)."""
    pass


@dataclass(frozen=True, eq=False)
class CommOp(OpNode):
    """Base class for communication operations (uses network/NVLink)."""
    pass


# -------- helpers --------

def tensor_replace(t: Tensor, **kwargs) -> Tensor:
    """dataclasses.replace wrapper (keeps call-sites clean)."""
    return replace(t, **kwargs)

def token_replace(tok: Token, **kwargs) -> Token:
    return replace(tok, **kwargs)