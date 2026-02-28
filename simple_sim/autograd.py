# name=simple_sim/autograd.py
from __future__ import annotations

from dataclasses import replace as dc_replace
from typing import Dict, List, Optional, Sequence, Tuple

from .ir import OpNode, Tensor, Token, _id_counter
from .extract import get_graph, topo_sort
from .ops_compute import add, ones_like
from .ops_schedule import DetachOp, wait_for, recompute_trigger
from .checkpoint import CheckpointStartOp, CheckpointEndOp, get_checkpoint_region


# ---------------------------------------------------------------------------
# Gradient marker helper
# ---------------------------------------------------------------------------

def _mark_gradient(t: Tensor) -> Tensor:
    """Stamp *is_gradient=True* on a tensor (for visualization)."""
    if not t.is_gradient:
        object.__setattr__(t, "is_gradient", True)
    return t


# ---------------------------------------------------------------------------
# Clone / replay helpers
# ---------------------------------------------------------------------------

def _clone_region(
    region_nodes: Tuple[OpNode, ...],
    start_tensor: Tensor,
    start_input: Tensor,
    end_input_tensor: Tensor,
) -> Tuple[List[OpNode], Dict[int, Tensor]]:
    """
    Clone the interior forward ops of a checkpoint region so that the
    replayed computation appears as fresh nodes in the backward graph.

    Parameters
    ----------
    region_nodes : tuple of OpNode
        Interior forward ops (excluding CheckpointStartOp / CheckpointEndOp).
    start_tensor : Tensor
        Output of CheckpointStartOp (enters the region).
    start_input : Tensor
        Tensor *before* CheckpointStartOp – recomputation feeds from here.
    end_input_tensor : Tensor
        Tensor fed into CheckpointEndOp (region's final output).

    Returns
    -------
    cloned_nodes : list[OpNode]   – topo-sorted cloned ops (fresh ids).
    tensor_remap : dict[id(old_tensor) -> new_tensor]
    """
    sorted_nodes = topo_sort(list(region_nodes))
    region_node_set = {id(n) for n in sorted_nodes}

    # --- 1. Discover every Tensor produced by a region node ----------------
    #     T is "produced by" node N iff T.producer is N.
    #     We find them by scanning inputs of downstream region nodes + the
    #     tensor going into CheckpointEndOp.
    produced_tensors: Dict[int, Tensor] = {}   # id(tensor) -> tensor

    def _scan(t: Tensor) -> None:
        if t.producer is not None and id(t.producer) in region_node_set:
            produced_tensors[id(t)] = t

    for n in sorted_nodes:
        for v in n.inputs:
            if isinstance(v, Tensor):
                _scan(v)
    _scan(end_input_tensor)

    # --- 2. Build remap table & clone nodes --------------------------------
    tensor_remap: Dict[int, Tensor] = {}
    tensor_remap[id(start_tensor)] = start_input   # region entry

    cloned: List[OpNode] = []

    for node in sorted_nodes:
        # Remap inputs
        new_inputs: list = []
        for v in node.inputs:
            if isinstance(v, Tensor) and id(v) in tensor_remap:
                new_inputs.append(tensor_remap[id(v)])
            else:
                new_inputs.append(v)          # external (e.g. parameter)

        # Clone the node with a fresh id and remapped inputs.
        # Only pass 'name' if the op class actually has a name field
        # (e.g. CheckpointStartOp/CheckpointEndOp do, but MatMulOp does not).
        replace_kwargs: dict = {
            "inputs": tuple(new_inputs),
            "id": next(_id_counter),
        }
        if hasattr(node, "name"):
            old_name = getattr(node, "name", None) or node.kind
            replace_kwargs["name"] = f"recompute/{old_name}"

        clone = dc_replace(node, **replace_kwargs)
        cloned.append(clone)

        # Create cloned output tensors for this node
        for tid, t in produced_tensors.items():
            if t.producer is not node:
                continue
            if tid in tensor_remap:
                continue
            cloned_t = Tensor(
                shape=t.shape,
                dtype=t.dtype,
                producer=clone,
                requires_grad=t.requires_grad,
                name=f"recompute/{t.name}" if t.name else None,
                shard=t.shard,
                tp_group=t.tp_group,
                dp_group=t.dp_group,
            )
            tensor_remap[tid] = cloned_t

    return cloned, tensor_remap


# ---------------------------------------------------------------------------
# Main backward pass
# ---------------------------------------------------------------------------

def backward(
    loss: Tensor,
    *,
    wrt: Sequence[Tensor],
    grad_output: Optional[Tensor] = None,
    keep_token: Optional[Token] = None,
) -> Dict[Tensor, Tensor]:
    """
    Minimal reverse-mode symbolic autograd.

    Each op defines its own ``vjp`` method.  Gradient checkpointing is
    supported: when the traversal encounters a ``CheckpointEndOp`` it
    clones & replays the stored forward sub-graph, then differentiates
    through the *replayed* ops so that the backward graph explicitly
    contains the recomputation cost.

    Args:
        loss: The tensor to differentiate (treated as the root).
        wrt: Tensors to compute gradients for.
        grad_output: Optional initial gradient for *loss*.  When ``None``
            (default), ``ones_like(loss)`` is used.  Provide an explicit
            tensor when the upstream gradient comes from an external
            source (e.g. a recv'd gradient in pipeline parallelism).
        keep_token: Optional token whose producer is included in the
            forward graph traversal (for scheduling dependencies).
    """
    roots: List[object] = [loss]
    if grad_output is not None:
        roots.append(grad_output)
    if keep_token is not None:
        roots.append(keep_token)

    fwd = get_graph(*roots, stop_at_detach=True)
    fwd_nodes = topo_sort(fwd.nodes)

    grad: Dict[Tensor, Tensor] = {}
    if grad_output is not None:
        grad[loss] = _mark_gradient(grad_output)
    else:
        grad[loss] = _mark_gradient(ones_like(loss, name="dLoss"))

    # node-id -> output tensors (stop at DetachOp & CheckpointStartOp)
    node_to_outputs: Dict[int, List[Tensor]] = {n.id: [] for n in fwd_nodes}

    def collect_tensors(roots: List[object]) -> None:
        visited: set = set()
        stack = list(roots)
        while stack:
            item = stack.pop()
            if id(item) in visited:
                continue
            visited.add(id(item))
            if isinstance(item, Tensor):
                if item.producer is not None and item.producer.id in node_to_outputs:
                    node_to_outputs[item.producer.id].append(item)
                    if isinstance(item.producer, DetachOp):
                        continue
                    if isinstance(item.producer, CheckpointStartOp):
                        continue
                    for inp in item.producer.inputs:
                        stack.append(inp)

    collect_tensors(roots)

    # Reverse-mode iteration
    for node in reversed(fwd_nodes):
        outs = node_to_outputs.get(node.id, [])
        if not outs:
            continue
        if len(outs) != 1:          # single-output ops only (v0)
            continue
        y = outs[0]
        dy = grad.get(y)
        if dy is None:
            continue

        # --- Checkpoint handling ------------------------------------------
        if isinstance(node, CheckpointEndOp):
            _backward_checkpoint(node, dy, grad)
            continue                 # CheckpointEndOp is identity; done.

        # --- Normal op ----------------------------------------------------
        input_grads = node.vjp(dy)
        for inp, g in input_grads.items():
            _mark_gradient(g)
            if inp in grad:
                grad[inp] = _mark_gradient(add(grad[inp], g, name=f"acc_d({inp.name})"))
            else:
                grad[inp] = g

    return {p: grad[p] for p in wrt if p in grad}


# ---------------------------------------------------------------------------
# Checkpoint backward: clone -> replay forward -> differentiate replay
# ---------------------------------------------------------------------------

def _find_checkpoint_start(
    end_node: CheckpointEndOp,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """
    Walk backward from *end_node*'s input through producer links to find
    the ``CheckpointStartOp`` with matching ``checkpoint_id``.

    This is context-aware: if *end_node* is a clone produced during an
    outer region's replay, the walk finds the *cloned* start node and
    returns the correct (recomputed) ``start_input``.

    Returns ``(start_output, start_input)`` or ``(None, None)``.
    """
    target_id = end_node.checkpoint_id
    visited: set = set()
    stack: List[Tensor] = [
        v for v in end_node.inputs if isinstance(v, Tensor)
    ]
    while stack:
        t = stack.pop()
        if id(t) in visited:
            continue
        visited.add(id(t))
        if t.producer is None:
            continue
        if (isinstance(t.producer, CheckpointStartOp)
                and t.producer.checkpoint_id == target_id):
            tinps = [v for v in t.producer.inputs if isinstance(v, Tensor)]
            return (t, tinps[0]) if tinps else (t, None)
        for inp in t.producer.inputs:
            if isinstance(inp, Tensor):
                stack.append(inp)
    return None, None


def _backward_checkpoint(
    end_node: CheckpointEndOp,
    grad_output: Tensor,
    grad: Dict[Tensor, Tensor],
) -> None:
    """
    Handle backward through a checkpoint region.

    1. Retrieve the forward sub-graph stored on ``end_node.region_nodes``.
    2. Clone all interior ops with fresh ids ("recompute/…" names).
    3. Build a local node_to_outputs for the cloned sub-graph.
    4. Differentiate through the cloned ops in reverse topo order,
       recursing into nested ``CheckpointEndOp`` clones.
    5. Merge resulting gradients (params + start-input) into *grad*.
    """
    region_nodes: tuple = end_node.region_nodes

    # --- Locate the *original* start tensor from the registry. -------------
    # region_nodes reference original graph nodes, so the remap key in
    # _clone_region must be the *original* start_tensor identity.
    region = get_checkpoint_region(end_node.checkpoint_id)
    if region is None or region.start_tensor is None:
        (x,) = [v for v in end_node.inputs if isinstance(v, Tensor)]
        _accum(grad, x, grad_output)
        return

    original_start_tensor = region.start_tensor

    # --- Find context-appropriate start_input. -----------------------------
    # Walking backward from end_node (which may be a clone created during an
    # outer region's replay) gives us the start_input that lives in the
    # correct graph context.
    _, start_input = _find_checkpoint_start(end_node)
    if start_input is None:
        # Fallback: derive from original start node.
        start_node = original_start_tensor.producer
        if start_node is None:
            (x,) = [v for v in end_node.inputs if isinstance(v, Tensor)]
            _accum(grad, x, grad_output)
            return
        (start_input,) = [v for v in start_node.inputs if isinstance(v, Tensor)]

    if not region_nodes:
        _accum(grad, start_input, grad_output)
        return

    # --- Gate recomputation on backward reaching this point. ---------------
    # Create a Token from grad_output so that the cloned forward ops
    # cannot be scheduled until backward has produced this gradient.
    trigger_tok = recompute_trigger(
        grad_output, name=f"recompute_trigger/{end_node.checkpoint_id}",
    )
    guarded_start = wait_for(
        start_input, trigger_tok,
        name=f"recompute_start/{start_input.name}" if start_input.name else None,
    )

    # --- Get original end_input_tensor. ------------------------------------
    # _clone_region's _scan() checks whether a tensor's producer is in
    # region_node_set (original node identities).  We must therefore pass
    # the *original* end_input_tensor so that its producer resolves correctly.
    original_end_tensor = region.end_tensor
    original_end_node = original_end_tensor.producer
    original_end_input = [
        v for v in original_end_node.inputs if isinstance(v, Tensor)
    ][0]

    # 1. Clone the forward region
    cloned_nodes, tensor_remap = _clone_region(
        region_nodes, original_start_tensor, guarded_start, original_end_input,
    )

    # 2. Build clone node_to_outputs
    cloned_id_set = {n.id for n in cloned_nodes}
    clone_to_outputs: Dict[int, List[Tensor]] = {n.id: [] for n in cloned_nodes}
    for tid, cloned_t in tensor_remap.items():
        if cloned_t.producer is not None and cloned_t.producer.id in cloned_id_set:
            clone_to_outputs[cloned_t.producer.id].append(cloned_t)

    # 3. Seed gradient at the replayed region output
    region_grad: Dict[Tensor, Tensor] = {}
    cloned_end_input = tensor_remap.get(id(original_end_input))
    if cloned_end_input is not None:
        region_grad[cloned_end_input] = grad_output
    else:
        _accum(grad, start_input, grad_output)
        return

    # 4. Backward through the cloned sub-graph
    for cnode in reversed(cloned_nodes):
        c_outs = clone_to_outputs.get(cnode.id, [])
        if not c_outs or len(c_outs) != 1:
            continue
        cy = c_outs[0]
        cdy = region_grad.get(cy)
        if cdy is None:
            continue

        # Recurse into nested checkpoint regions
        if isinstance(cnode, CheckpointEndOp):
            _backward_checkpoint(cnode, cdy, region_grad)
            continue

        input_grads = cnode.vjp(cdy)
        for inp, g in input_grads.items():
            _mark_gradient(g)
            if inp in region_grad:
                region_grad[inp] = _mark_gradient(add(region_grad[inp], g,
                                       name=f"acc_d({inp.name})"))
            else:
                region_grad[inp] = g

    # 5. Merge external gradients into caller's grad dict.
    #    Only skip tensors *produced by* cloned nodes (internal to replay).
    #    start_input is a remap VALUE but not produced by any clone — its
    #    gradient must flow back to the caller.
    remap_id_set: set = set()
    for ct in tensor_remap.values():
        if ct.producer is not None and ct.producer.id in cloned_id_set:
            remap_id_set.add(id(ct))

    for t, g in region_grad.items():
        if id(t) in remap_id_set:
            continue                 # internal to replay
        _accum(grad, t, g)


def _accum(grad: Dict[Tensor, Tensor], t: Tensor, g: Tensor) -> None:
    """Accumulate gradient *g* for tensor *t* in *grad*."""
    _mark_gradient(g)
    if t in grad:
        grad[t] = _mark_gradient(add(grad[t], g, name=f"acc_d({t.name})"))
    else:
        grad[t] = g
