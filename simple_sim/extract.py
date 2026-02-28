from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Set, Tuple, Union

from .ir import MemoryCategory, OpNode, Tensor, Token, Value

Root = Union[Tensor, Token]

# Import here to avoid circular import at module level
_detach_op_class = None

def _get_detach_op_class():
    global _detach_op_class
    if _detach_op_class is None:
        from .ops_schedule import DetachOp
        _detach_op_class = DetachOp
    return _detach_op_class

@dataclass
class ExtractedGraph:
    nodes: List[OpNode]
    roots: Tuple[Root, ...]

def get_graph(*roots: Root, stop_at_detach: bool = False) -> ExtractedGraph:
    """
    Extract computation graph from root tensors/tokens.
    
    Args:
        roots: Root values to trace from
        stop_at_detach: If True, don't traverse inputs of DetachOp nodes.
                        Useful for backward pass to prevent tracing into
                        previous iterations.
    """
    visited: Set[int] = set()
    nodes: Dict[int, OpNode] = {}
    stack: List[OpNode] = []
    
    DetachOp = _get_detach_op_class() if stop_at_detach else None

    def push_val(v: Root) -> None:
        if isinstance(v, Tensor):
            if v.producer is not None:
                stack.append(v.producer)
        else:
            stack.append(v.producer)

    for r in roots:
        push_val(r)

    while stack:
        n = stack.pop()
        if n.id in visited:
            continue
        visited.add(n.id)
        nodes[n.id] = n
        
        # Stop traversal at DetachOp if requested
        if stop_at_detach and isinstance(n, DetachOp):
            continue
            
        for inp in n.inputs:
            if isinstance(inp, Tensor):
                if inp.producer is not None:
                    stack.append(inp.producer)
            else:
                stack.append(inp.producer)

    out = [nodes[k] for k in sorted(nodes.keys())]
    return ExtractedGraph(nodes=out, roots=tuple(roots))

def topo_sort(nodes: Iterable[OpNode]) -> List[OpNode]:
    nodes = list(nodes)
    by_id = {n.id: n for n in nodes}
    indeg = {n.id: 0 for n in nodes}
    succ: Dict[int, List[int]] = {n.id: [] for n in nodes}

    def deps(n: OpNode) -> List[int]:
        out: List[int] = []
        for v in n.inputs:
            prod = None
            if isinstance(v, Tensor):
                prod = v.producer
            else:
                prod = v.producer
            if prod is not None and prod.id in indeg:
                out.append(prod.id)
        return out

    for n in nodes:
        for d in deps(n):
            indeg[n.id] += 1
            succ[d].append(n.id)

    q = sorted([nid for nid, deg in indeg.items() if deg == 0])
    out: List[OpNode] = []
    while q:
        nid = q.pop(0)
        out.append(by_id[nid])
        for sid in succ[nid]:
            indeg[sid] -= 1
            if indeg[sid] == 0:
                q.append(sid)
                q.sort()
    return out


# ---------------------------------------------------------------------------
# Activation memory analysis  (trace-based, release-after-last-use)
# ---------------------------------------------------------------------------

def _tensor_bytes(t: Tensor) -> int:
    from .utils import bytes_of
    return bytes_of(t.physical_shape(), t.dtype)


def _resolve_memory_owner(t: Tensor) -> Tensor:
    """Follow alias chain to the tensor that actually owns memory."""
    while t.aliases is not None:
        t = t.aliases
    return t


@dataclass
class ActivationMemoryInfo:
    """Result of ``get_activation_summary``.

    Fields
    ------
    peak_bytes : int
        Peak activation memory during execution (release-after-last-use).
    peak_step : int
        Topo-order step index at which peak memory occurs.
    peak_node : OpNode | None
        The node whose execution triggered the peak.
    timeline : List[Tuple[int, int]]
        ``(step, current_bytes)`` after each node is executed.
    live_at_peak : List[Tensor]
        Memory-owning tensors that are live at the peak step.
    parameter_tensors : List[Tensor]
        Trainable weights (no producer in graph).  Not counted towards
        activation peak (they live for the entire lifetime).
    input_tensors : List[Tensor]
        Graph inputs (no producer).  Not counted towards activation peak.
    """
    peak_bytes: int
    peak_step: int
    peak_node: OpNode | None
    timeline: List[Tuple[int, int]]
    live_at_peak: List[Tensor]
    parameter_tensors: List[Tensor]
    input_tensors: List[Tensor]


def get_activation_summary(graph: ExtractedGraph) -> ActivationMemoryInfo:
    """
    Trace through *graph* in topo order, simulating a release-after-last-use
    memory policy for all MATERIALIZED tensors.

    For every MATERIALIZED tensor we:
      1. **Allocate** it when its producing node executes.
      2. **Release** it right after its last consumer executes.

    NOT_MATERIALIZED tensors (zero-cost views / aliases) are not tracked
    individually -- their memory is attributed to the root owner via
    ``_resolve_memory_owner``.

    Parameters and graph inputs (tensors with no producer inside the graph)
    are assumed to be permanently resident and are reported separately.
    They are **not** included in the peak activation count.

    Returns an ``ActivationMemoryInfo`` with the peak, timeline, etc.
    """
    sorted_nodes = topo_sort(graph.nodes)
    node_ids = {n.id for n in sorted_nodes}

    # --- 1. Discover every unique Tensor that appears as an op input. ------
    seen_tensors: Set[int] = set()           # id(tensor)
    all_tensors: List[Tensor] = []
    for node in sorted_nodes:
        for v in node.inputs:
            if isinstance(v, Tensor) and id(v) not in seen_tensors:
                seen_tensors.add(id(v))
                all_tensors.append(v)

    # --- 2. Classify tensors. ----------------------------------------------
    params: List[Tensor] = []
    inputs: List[Tensor] = []

    # Materialized tensors that need tracking (produced inside the graph and
    # actually own memory -- not aliases).
    # Key: id(memory_owner_tensor)  Value: memory_owner Tensor
    tracked: Dict[int, Tensor] = {}

    # Map id(memory_owner) -> set of node-ids that consume it
    # (either directly or through an alias chain).
    consumers: Dict[int, Set[int]] = {}

    for t in all_tensors:
        has_producer = t.producer is not None and t.producer.id in node_ids
        if not has_producer:
            # External tensor (parameter or input).
            if t.requires_grad:
                params.append(t)
            else:
                inputs.append(t)
            continue

        # Resolve the memory owner for this tensor.
        owner = _resolve_memory_owner(t)
        owner_key = id(owner)

        # Only track if the owner is MATERIALIZED and produced inside graph.
        if owner.memory_category is not MemoryCategory.MATERIALIZED:
            continue
        if owner.producer is None or owner.producer.id not in node_ids:
            continue

        tracked[owner_key] = owner
        if owner_key not in consumers:
            consumers[owner_key] = set()

    # Also scan root tensors to discover materialized tensors that are *only*
    # graph outputs and never appear as an input to another node.
    for root in graph.roots:
        if not isinstance(root, Tensor):
            continue
        owner = _resolve_memory_owner(root)
        owner_key = id(owner)
        if owner.memory_category is not MemoryCategory.MATERIALIZED:
            continue
        if owner.producer is None or owner.producer.id not in node_ids:
            continue
        if owner_key not in tracked:
            tracked[owner_key] = owner
        if owner_key not in consumers:
            consumers[owner_key] = set()

    # --- 3. Build consumer sets (which nodes consume each tracked tensor). --
    for node in sorted_nodes:
        for v in node.inputs:
            if not isinstance(v, Tensor):
                continue
            owner = _resolve_memory_owner(v)
            owner_key = id(owner)
            if owner_key in consumers:
                consumers[owner_key].add(node.id)

    # --- 4. Build producer map: id(owner) -> producing node id. ------------
    producer_of: Dict[int, int] = {}
    for owner_key, owner in tracked.items():
        if owner.producer is not None:
            producer_of[owner_key] = owner.producer.id

    # --- 5. Simulate execution in topo order. ------------------------------
    node_order: Dict[int, int] = {n.id: i for i, n in enumerate(sorted_nodes)}

    # last_use[owner_key] = node-id of last consumer (by topo position).
    last_use: Dict[int, int] = {}
    for owner_key, cons in consumers.items():
        if cons:
            last_use[owner_key] = max(cons, key=lambda nid: node_order[nid])
        else:
            # Produced but never consumed (graph-output root).
            # Keep alive until the very last node.
            last_use[owner_key] = sorted_nodes[-1].id

    current_bytes = 0
    peak_bytes = 0
    peak_step = 0
    peak_node: OpNode | None = None
    live: Set[int] = set()                   # owner_keys currently live
    timeline: List[Tuple[int, int]] = []

    for step, node in enumerate(sorted_nodes):
        # Allocate: tensors produced by this node.
        for owner_key, prod_nid in producer_of.items():
            if prod_nid == node.id and owner_key not in live:
                live.add(owner_key)
                current_bytes += _tensor_bytes(tracked[owner_key])

        # Track peak.
        if current_bytes > peak_bytes:
            peak_bytes = current_bytes
            peak_step = step
            peak_node = node

        timeline.append((step, current_bytes))

        # Release: tensors whose last consumer is this node.
        to_release: List[int] = []
        for owner_key in live:
            if last_use.get(owner_key) == node.id:
                to_release.append(owner_key)
        for owner_key in to_release:
            live.discard(owner_key)
            current_bytes -= _tensor_bytes(tracked[owner_key])

    # Snapshot of live tensors at peak.
    live_at_peak: List[Tensor] = []
    if peak_node is not None:
        sim_live: Set[int] = set()
        for step, node in enumerate(sorted_nodes):
            for owner_key, prod_nid in producer_of.items():
                if prod_nid == node.id and owner_key not in sim_live:
                    sim_live.add(owner_key)
            if step == peak_step:
                live_at_peak = [tracked[k] for k in sim_live]
                break
            to_release = [
                k for k in sim_live if last_use.get(k) == node.id
            ]
            for k in to_release:
                sim_live.discard(k)

    return ActivationMemoryInfo(
        peak_bytes=peak_bytes,
        peak_step=peak_step,
        peak_node=peak_node,
        timeline=timeline,
        live_at_peak=live_at_peak,
        parameter_tensors=params,
        input_tensors=inputs,
    )