"""Gradient checkpointing infrastructure.

Provides the mechanism to mark regions of the forward graph for
recomputation during backward, trading compute for memory.

Contents:
- CheckpointRegion          — bookkeeping for a single region
- CheckpointStartOp / checkpoint_start — region entry marker
- CheckpointEndOp  / checkpoint_end   — region exit marker (eagerly traces)
- checkpoint()              — context-manager API
- clear_checkpoint_registry — reset between independent graphs
- get_checkpoint_region / get_all_checkpoint_regions — queries
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Dict

from .ir import ComputeOp, CostMeta, MemoryCategory, OpNode, Tensor


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_checkpoint_registry: Dict[int, "CheckpointRegion"] = {}
_checkpoint_id_counter = 0


class CheckpointRegion:
    """Tracks a checkpoint region during graph construction."""
    def __init__(self, checkpoint_id: int, name: str):
        self.checkpoint_id = checkpoint_id
        self.name = name
        self.start_tensor: Tensor | None = None
        self.end_tensor: Tensor | None = None
        self.nodes_in_region: list = []  # Nodes between start and end


# ---------------------------------------------------------------------------
# Ops
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class CheckpointStartOp(ComputeOp):
    """
    Marks the start of a gradient checkpoint region.

    During forward: acts as identity (pass-through)
    During backward: signals to recompute the region instead of using saved activations
    """
    checkpoint_id: int = 0
    name: str = "checkpoint_start"
    kind: str = field(default="checkpoint_start", init=False)

    def get_cost_meta(self) -> CostMeta:
        # Identity operation - no cost
        return {"flops": 0, "mem_read": 0, "mem_write": 0}

    def vjp(self, grad_output: Tensor) -> Dict[Tensor, Tensor]:
        # Pass through gradient
        (x,) = [v for v in self.inputs if isinstance(v, Tensor)]
        if not x.requires_grad:
            return {}
        return {x: grad_output}


@dataclass(frozen=True, eq=False)
class CheckpointEndOp(ComputeOp):
    """
    Marks the end of a gradient checkpoint region.

    During forward: acts as identity (pass-through)
    During backward: triggers recomputation of the checkpoint region

    Stores the region_nodes (list of ops between start and end) for replay during backward.
    """
    checkpoint_id: int = 0
    name: str = "checkpoint_end"
    kind: str = field(default="checkpoint_end", init=False)
    region_nodes: tuple = field(default_factory=tuple)  # Store forward ops for recomputation

    def get_cost_meta(self) -> CostMeta:
        # Identity operation - no cost
        return {"flops": 0, "mem_read": 0, "mem_write": 0}

    def vjp(self, grad_output: Tensor) -> Dict[Tensor, Tensor]:
        # Pass through gradient - actual recomputation handled in backward()
        (x,) = [v for v in self.inputs if isinstance(v, Tensor)]
        if not x.requires_grad:
            return {}
        return {x: grad_output}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def checkpoint_start(x: Tensor, checkpoint_id: int, name: str) -> Tensor:
    """
    Mark the start of a gradient checkpoint region.

    Args:
        x: Input tensor
        checkpoint_id: Unique identifier for this checkpoint region
        name: Name for debugging

    Returns:
        Tensor that passes through x (identity operation)
    """
    node = CheckpointStartOp(checkpoint_id=checkpoint_id, name=name, inputs=(x,))
    out = Tensor(
        shape=x.shape,
        dtype=x.dtype,
        producer=node,
        requires_grad=x.requires_grad,
        name=name or x.name,
        tp_group=x.tp_group,
        dp_group=x.dp_group,
        shard=x.shard,
        memory_category=MemoryCategory.NOT_MATERIALIZED,
        aliases=x,
    )

    # Register in global registry
    if checkpoint_id not in _checkpoint_registry:
        _checkpoint_registry[checkpoint_id] = CheckpointRegion(checkpoint_id, name)
    _checkpoint_registry[checkpoint_id].start_tensor = out

    return out


def _trace_region_nodes(end_input: Tensor, checkpoint_id: int) -> tuple:
    """
    Walk backward from *end_input* through Tensor.producer links and collect
    every OpNode that belongs to the checkpoint region identified by
    *checkpoint_id*.  The start and end checkpoint ops themselves are
    **excluded** – only the interior compute/comm nodes are returned.

    A node is considered "inside" the region only if its id is strictly
    greater than the matching CheckpointStartOp's id.  This prevents
    the traversal from following external inputs (parameters, tensors from
    prior iterations) whose producers live outside the region.

    Returns a tuple of OpNode (no guaranteed order – caller should topo-sort).
    """
    # Look up the start node to get the id boundary.
    region = _checkpoint_registry.get(checkpoint_id)
    if region is None or region.start_tensor is None:
        return ()
    start_node = region.start_tensor.producer
    if start_node is None:
        return ()
    start_node_id = start_node.id

    collected: dict[int, OpNode] = {}  # id -> node
    visited: set[int] = set()
    stack: list[OpNode] = []

    # Seed with the producer of the tensor fed into CheckpointEndOp
    if end_input.producer is not None:
        stack.append(end_input.producer)

    while stack:
        node = stack.pop()
        if node.id in visited:
            continue
        visited.add(node.id)

        # If we hit the matching CheckpointStartOp, stop – don't include it
        if isinstance(node, CheckpointStartOp) and node.checkpoint_id == checkpoint_id:
            continue

        # Only collect nodes created *after* the start op (inside the region).
        # External nodes (parameters, prior-iteration ops) have earlier ids.
        if node.id <= start_node_id:
            continue

        collected[node.id] = node

        for inp in node.inputs:
            if isinstance(inp, Tensor) and inp.producer is not None:
                stack.append(inp.producer)

    return tuple(collected.values())


def checkpoint_end(x: Tensor, checkpoint_id: int, name: str) -> Tensor:
    """
    Mark the end of a gradient checkpoint region.

    At construction time we eagerly trace the forward sub-graph between
    the matching CheckpointStartOp and this node, and store the interior
    nodes on ``region_nodes`` so the backward pass can clone & replay them.

    Args:
        x: Input tensor (output of the checkpoint region)
        checkpoint_id: Unique identifier matching the corresponding checkpoint_start
        name: Name for debugging

    Returns:
        Tensor that passes through x (identity operation)
    """
    # Eagerly trace the interior nodes of the region
    region_nodes = _trace_region_nodes(x, checkpoint_id)

    node = CheckpointEndOp(
        checkpoint_id=checkpoint_id,
        name=name,
        inputs=(x,),
        region_nodes=region_nodes,
    )
    out = Tensor(
        shape=x.shape,
        dtype=x.dtype,
        producer=node,
        requires_grad=x.requires_grad,
        name=name or x.name,
        tp_group=x.tp_group,
        dp_group=x.dp_group,
        shard=x.shard,
        memory_category=MemoryCategory.NOT_MATERIALIZED,
        aliases=x,
    )
    # The output of checkpoint_end is a zero-cost view aliasing the region's
    # exit tensor. Clear the checkpoint tag since this tensor flows OUT of
    # the region.
    object.__setattr__(out, "checkpoint_region_id", None)

    # Register in global registry
    if checkpoint_id in _checkpoint_registry:
        region = _checkpoint_registry[checkpoint_id]
        region.end_tensor = out
        region.nodes_in_region = list(region_nodes)

    return out


# ---------------------------------------------------------------------------
# Context-manager API
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def checkpoint(*, name: str = "checkpoint"):
    """
    Context manager for gradient checkpointing.

    Usage:
        with checkpoint(name="layer1") as ckpt:
            h = ckpt.enter(x)   # marks start of checkpoint region
            h = layer1(h)
            h = layer2(h)
            h = layer3(h)
            h = ckpt.exit(h)   # marks end of checkpoint region
        output = layer4(h)

    During backward, the region between checkpoint_start and checkpoint_end
    will be recomputed instead of using saved activations.

    Supports nested checkpoints - each with checkpoint() creates a new region.
    """
    global _checkpoint_id_counter

    # Allocate a new checkpoint ID
    checkpoint_id = _checkpoint_id_counter
    _checkpoint_id_counter += 1

    # Create registry entry
    region = CheckpointRegion(checkpoint_id, name)
    _checkpoint_registry[checkpoint_id] = region

    class CheckpointContext:
        """Helper to track and wrap tensors within the checkpoint region."""
        def __init__(self, ckpt_id, ckpt_name):
            self.checkpoint_id = ckpt_id
            self.name = ckpt_name

        def enter(self, x: Tensor) -> Tensor:
            """Mark the start of the checkpoint region with input tensor."""
            return checkpoint_start(x, self.checkpoint_id, f"{self.name}.start")

        def exit(self, x: Tensor) -> Tensor:
            """Mark the end of the checkpoint region with output tensor."""
            return checkpoint_end(x, self.checkpoint_id, f"{self.name}.end")

    ctx = CheckpointContext(checkpoint_id, name)

    # Push the checkpoint ID so that all Tensors created inside this block
    # are automatically tagged with ``checkpoint_region_id``.
    from .ir import _active_checkpoint_stack
    _active_checkpoint_stack.append(checkpoint_id)
    try:
        yield ctx
    finally:
        _active_checkpoint_stack.pop()


# ---------------------------------------------------------------------------
# Registry queries / reset
# ---------------------------------------------------------------------------

def clear_checkpoint_registry():
    """Clear the checkpoint registry. Call between independent graphs."""
    global _checkpoint_registry, _checkpoint_id_counter
    _checkpoint_registry = {}
    _checkpoint_id_counter = 0
    # Also clear the active stack (safety net for interrupted contexts).
    from .ir import _active_checkpoint_stack
    _active_checkpoint_stack.clear()


def get_checkpoint_region(checkpoint_id: int) -> CheckpointRegion | None:
    """Get a checkpoint region by ID."""
    return _checkpoint_registry.get(checkpoint_id)


def get_all_checkpoint_regions() -> Dict[int, CheckpointRegion]:
    """Get all registered checkpoint regions."""
    return _checkpoint_registry.copy()
