"""Scheduling primitives — zero-cost graph-structural ops.

These ops carry no computational cost; they exist purely to express
data-flow / ordering constraints in the computation graph.

Contents:
- DetachOp / detach        — stop gradient flow
- WaitForOp / wait_for     — tensor + token scheduling barrier
- CompletionOp / completion — tensor(s) → token barrier
- RecomputeTriggerOp / recompute_trigger — gradient → token gate
- param_factory / input_factory — cross-iteration chaining helpers
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence

from .ir import ComputeOp, CostMeta, MemoryCategory, Tensor, Token


# -------- DetachOp --------

@dataclass(frozen=True, eq=False)
class DetachOp(ComputeOp):
    """
    Stop-gradient operation. Creates a new tensor that looks like the input
    but blocks gradient flow during backward pass.

    Similar to PyTorch's detach() or JAX's stop_gradient().
    This is essential for multi-iteration training where we want to use
    updated parameters without tracing gradients back through previous iterations.
    """
    kind: str = field(default="detach", init=False)

    def get_cost_meta(self) -> CostMeta:
        return {"flops": 0, "mem_read": 0, "mem_write": 0}

    def vjp(self, grad_output: Tensor) -> Dict[Tensor, Tensor]:
        # Stop gradient - return no gradients to input
        return {}


def detach(x: Tensor, *, name: str | None = None) -> Tensor:
    """
    Create a tensor that stops gradient flow.

    The output tensor has the same values/shape as input, but backward
    will not propagate gradients through this operation.

    Use this when chaining iterations to prevent infinite backward tracing.

    The tensor IS part of the forward graph (has a producer), but backward
    traversal will stop here because:
    1. DetachOp.vjp() returns {} (no gradients to inputs)
    2. backward's collect_tensors stops at DetachOp nodes
    """
    node = DetachOp(inputs=(x,))
    out = Tensor(
        shape=x.shape,
        dtype=x.dtype,
        producer=node,  # Part of forward graph
        requires_grad=True,  # Can still receive gradients in this iteration
        name=name or x.name,
        tp_group=x.tp_group,
        dp_group=x.dp_group,
        shard=x.shard,
        memory_category=MemoryCategory.NOT_MATERIALIZED,
        aliases=x,
    )
    return out


# -------- WaitForOp --------

@dataclass(frozen=True, eq=False)
class WaitForOp(ComputeOp):
    """
    Scheduling barrier: tensor output depends on both input tensor and a token.

    This creates an explicit dependency edge: the output tensor is only "available"
    after the token's producer completes. Useful for modeling:
    - Parameter availability after optimizer step completes
    - Synchronization points between iterations
    - Pipelining effects

    The op has no computational cost (pure scheduling).
    """
    kind: str = field(default="wait_for", init=False)

    def get_cost_meta(self) -> CostMeta:
        return {"flops": 0, "mem_read": 0, "mem_write": 0}

    def vjp(self, grad_output: Tensor) -> Dict[Tensor, Tensor]:
        # Pass through gradient unchanged
        (x,) = [v for v in self.inputs if isinstance(v, Tensor)]
        if not x.requires_grad:
            return {}
        return {x: grad_output}


def wait_for(x: Tensor, after: "Token | Tensor", *, name: str | None = None) -> Tensor:
    """
    Create a tensor that depends on both the input tensor and a dependency.

    This is a scheduling primitive - the output has the same value as input,
    but cannot be used until *after*'s producer completes.

    *after* can be a Token or a Tensor (e.g. the NOT_MATERIALIZED output
    of a send operation).
    """
    node = WaitForOp(inputs=(x, after))
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
    return out


# -------- CompletionOp --------

@dataclass(frozen=True, eq=False)
class CompletionOp(ComputeOp):
    """
    Scheduling-only op that converts one or more tensors into a Token.

    This creates a dependency edge: the Token is only available once all
    input tensors have been produced.  Useful for expressing "iteration N
    is done" or "all optimizer updates have finished".

    Zero cost — pure scheduling primitive.
    """
    kind: str = field(default="completion", init=False)

    def get_cost_meta(self) -> CostMeta:
        return {"flops": 0, "mem_read": 0, "mem_write": 0}

    def vjp(self, grad_output: Tensor) -> Dict[Tensor, Tensor]:
        return {}


def completion(*tensors: Tensor, name: str | None = None) -> Token:
    """
    Create a Token that becomes available once all *tensors* are ready.

    Use cases:
    - Mark the end of a training iteration
    - Gate the next iteration's inputs / parameters behind optimizer
      completion
    - Any place where you need a "barrier" token derived from tensors
    """
    node = CompletionOp(inputs=tuple(tensors))
    return Token(producer=node, name=name)


# -------- RecomputeTriggerOp --------

@dataclass(frozen=True, eq=False)
class RecomputeTriggerOp(ComputeOp):
    """
    Scheduling-only op that converts a gradient tensor into a Token.

    Used during backward to create a dependency edge: the recomputed
    forward sub-graph should not start until backward has actually
    reached the checkpoint boundary (i.e. ``grad_output`` is available).
    """
    kind: str = field(default="recompute_trigger", init=False)

    def get_cost_meta(self) -> CostMeta:
        return {"flops": 0, "mem_read": 0, "mem_write": 0}

    def vjp(self, grad_output: Tensor) -> Dict[Tensor, Tensor]:
        return {}  # no gradient through a scheduling token


def recompute_trigger(grad: Tensor, *, name: str | None = None) -> Token:
    """
    Create a Token that becomes available once *grad* is ready.

    This is used to gate recomputation in the backward pass so that
    cloned forward ops depend on backward actually reaching the
    checkpoint boundary rather than being schedulable immediately.
    """
    node = RecomputeTriggerOp(inputs=(grad,))
    return Token(producer=node, name=name)


# -------- Cross-iteration chaining helpers --------

def param_factory(
    prev_params: List[Tensor] | None,
    *,
    after: "Token | None" = None,
    create_fn: "Callable[[], List[Tensor]]",
    iteration: int = 0,
) -> List[Tensor]:
    """
    Create parameters for an iteration with flexible dependency modeling.

    This is the recommended way to handle parameter chaining between iterations.

    Args:
        prev_params: Parameters from previous iteration (None for first iteration)
        after: Token to wait for before params become available.
               - None: params available immediately after update (or creation)
               - Token: params only available after token's producer completes
        create_fn: Function to create fresh parameters (called only if prev_params is None)
        iteration: Current iteration number (for naming)

    Dependency modeling strategies:

    1. after=None (immediate availability):
       Params are available as soon as optimizer update finishes.
       Good for overlapping compute with next iteration's forward.

    2. after=prev_iteration_done_token:
       Params wait for full iteration completion before becoming available.
       Models stricter synchronization, useful for:
       - Pipeline barriers
       - All-reduce completion before next forward
       - Memory consistency points

    Returns:
        List of parameter tensors, properly detached and optionally with dependencies.
    """
    if prev_params is None:
        # First iteration: create fresh parameters
        return create_fn()

    result = []
    for p in prev_params:
        # Apply scheduling dependency if token provided
        if after is not None:
            p = wait_for(p, after, name=p.name)

        # Detach to stop gradient flow from tracing into previous iteration
        p = detach(p, name=f"iter{iteration}.{p.name}" if p.name else None)
        result.append(p)

    return result


def input_factory(
    prev_input: Tensor | None,
    *,
    after: "Token | None" = None,
    create_fn: "Callable[[], Tensor]",
    iteration: int = 0,
) -> Tensor:
    """
    Create (or reuse) an input tensor for an iteration.

    The underlying data tensor is materialised only once.  In subsequent
    iterations the same tensor is gated behind a ``wait_for`` dependency
    so that it appears connected in the graph rather than as a
    disconnected starting point.

    Args:
        prev_input: The input tensor from the previous iteration
                    (``None`` for the first iteration).
        after:      Optional Token the input should depend on (e.g. the
                    previous iteration's completion token).  Ignored on
                    the first iteration.
        create_fn:  Callable that creates the fresh input tensor.
                    Only called when ``prev_input is None``.
        iteration:  Current iteration number (for naming).

    Returns:
        A Tensor that can be used as the iteration's input.
    """
    if prev_input is None:
        return create_fn()

    # Re-use the same tensor, gated behind the previous iteration.
    t = prev_input
    if after is not None:
        t = wait_for(t, after, name=t.name)
    return t
