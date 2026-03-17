"""Chrome Trace Format visualization for multi-GPU computation graphs.

This module exports MultiRankGraph to Chrome Trace JSON format, viewable in:
- chrome://tracing (Chrome/Chromium)
- https://ui.perfetto.dev/ (Perfetto)

Features:
- Each GPU rank is a separate process (row in timeline)
- Duration events for compute/comm operations
- Flow events (arrows) for send→recv dependencies
- Aligned timestamps for collective operations

Usage:
    from simple_sim import aggregate_graphs, build_rank, get_graph
    from simple_sim.timeline_visualize import to_chrome_trace, schedule_multi_rank
    
    # Build and aggregate graphs
    graphs = {rank: build_graph_for_rank(rank) for rank in range(4)}
    multi = aggregate_graphs(graphs)
    
    # Export with automatic scheduling
    to_chrome_trace(multi, output="timeline.json")
    
    # Or with custom cost function
    def my_cost(node):
        return node.get_cost_meta()["flops"] / 1e9  # ns per GFLOP
    to_chrome_trace(multi, output="timeline.json", cost_fn=my_cost)
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple, TYPE_CHECKING

from .aggregate import MultiRankGraph, CrossRankCollective
from .extract import topo_sort
from .ir import CommOp, ComputeOp, OpNode
from .visualize import _get_display_kind, _get_node_color, _format_num

if TYPE_CHECKING:
    from .ops_comm import SendOp, FillOp


# -------- Scheduling-only ops (excluded from trace) --------

# These ops exist purely for graph scheduling/dependency resolution
# and don't represent actual compute or communication kernels
SCHEDULING_OP_KINDS = frozenset({
    "sink",           # Dependency aggregation (no actual work)
    "wait_for",       # Scheduling barrier (no actual work)
    "completion",     # Tensor → token conversion (no actual work)
    "detach",         # Stop gradient (no actual work)
    "recompute_trigger",  # Checkpoint trigger (no actual work)
})


def is_scheduling_op(node: OpNode) -> bool:
    """Return True if this op is purely for scheduling (no actual kernel)."""
    return node.kind in SCHEDULING_OP_KINDS


# -------- Cost functions --------

def default_cost_fn(node: OpNode) -> float:
    """Default cost function: returns duration in microseconds.
    
    Uses flops/mem from cost_meta, or falls back to 10µs per op.
    Scheduling ops return 0 (instant completion for dependency resolution).
    """
    # Scheduling-only ops complete instantly (just dependency edges)
    if is_scheduling_op(node):
        return 0.0
    
    try:
        cost = node.get_cost_meta()
        flops = cost.get("flops", 0)
        mem = cost.get("mem_read", 0) + cost.get("mem_write", 0)
        
        # Simple roofline: assume 1 TFLOP/s compute, 1 TB/s memory
        # Time = max(compute_time, memory_time)
        compute_us = flops / 1e6  # 1 TFLOP/s = 1e12 FLOP/s = 1e6 FLOP/µs
        memory_us = mem * 2 / 1e6  # 1 TB/s = 1e12 B/s, assume 2 bytes/element
        
        duration = max(compute_us, memory_us, 1.0)  # Minimum 1µs
        return duration
    except Exception:
        return 10.0  # Default 10µs


def topo_order_cost_fn(node: OpNode) -> float:
    """Simple cost function: fixed duration per op (for layout testing).
    
    Scheduling ops return 0 (instant completion).
    """
    if is_scheduling_op(node):
        return 0.0
    return 100.0  # 100µs per op


# -------- Scheduling --------

@dataclass
class ScheduledOp:
    """An operation with computed start/end times."""
    node: OpNode
    rank: int
    start_us: float
    end_us: float


def schedule_multi_rank(
    multi: MultiRankGraph,
    cost_fn: Callable[[OpNode], float] = default_cost_fn,
) -> Dict[int, List[ScheduledOp]]:
    """Schedule operations respecting dependencies.
    
    Computes start/end times for all operations, ensuring:
    1. Intra-rank dependencies: op starts after all inputs are ready
    2. Cross-rank send→recv: recv starts after send completes
    3. Collectives: all participants start at the same time (max of input ready times)
    
    Args:
        multi: MultiRankGraph with aggregated graphs
        cost_fn: Function mapping OpNode to duration in microseconds
    
    Returns:
        Dict mapping rank to list of ScheduledOp
    """
    # Build node ID to (rank, node) mapping
    node_by_id: Dict[int, Tuple[int, OpNode]] = {}
    for rank, g in multi.graphs.items():
        for node in g.nodes:
            node_by_id[node.id] = (rank, node)
    
    # Track end times for all nodes
    end_times: Dict[int, float] = {}  # node.id -> end_time
    start_times: Dict[int, float] = {}  # node.id -> start_time
    
    # Build send→recv mapping for cross-rank dependencies
    send_to_recv: Dict[int, int] = {}  # send_node.id -> recv_node.id
    recv_to_send: Dict[int, int] = {}  # recv_node.id -> send_node.id
    for send_op, recv_op, src_rank, dst_rank in multi.send_recv_pairs:
        send_to_recv[send_op.id] = recv_op.id
        recv_to_send[recv_op.id] = send_op.id
    
    # Build collective grouping: node.id -> CrossRankCollective
    node_to_collective: Dict[int, CrossRankCollective] = {}
    for coll in multi.cross_collectives:
        for rank, op in coll.participants:
            node_to_collective[op.id] = coll
    
    # Process each rank's graph in topological order
    # But we need a global ordering that respects cross-rank deps
    
    # Simple approach: iterate until all scheduled
    # More sophisticated: global topo sort considering cross-rank edges
    
    scheduled: Dict[int, List[ScheduledOp]] = {rank: [] for rank in multi.ranks}
    rank_current_time: Dict[int, float] = {rank: 0.0 for rank in multi.ranks}
    
    # Get topo-sorted nodes per rank
    topo_per_rank: Dict[int, List[OpNode]] = {}
    for rank, g in multi.graphs.items():
        topo_per_rank[rank] = topo_sort(g.nodes)
    
    # Track which nodes are scheduled
    scheduled_ids: set[int] = set()
    
    # Index into each rank's topo order
    rank_idx: Dict[int, int] = {rank: 0 for rank in multi.ranks}
    
    max_iterations = sum(len(nodes) for nodes in topo_per_rank.values()) * 2
    iteration = 0
    
    while len(scheduled_ids) < sum(len(nodes) for nodes in topo_per_rank.values()):
        iteration += 1
        if iteration > max_iterations:
            raise RuntimeError("Scheduling did not converge - possible cycle in dependencies")
        
        made_progress = False
        
        for rank in multi.ranks:
            nodes = topo_per_rank[rank]
            idx = rank_idx[rank]
            
            if idx >= len(nodes):
                continue
            
            node = nodes[idx]
            
            # Check if all dependencies are satisfied
            ready_time = rank_current_time[rank]
            can_schedule = True
            
            # Check intra-rank dependencies (inputs)
            for inp in node.inputs:
                if hasattr(inp, 'producer') and inp.producer is not None:
                    prod_id = inp.producer.id
                    if prod_id in node_by_id and prod_id not in scheduled_ids:
                        can_schedule = False
                        break
                    if prod_id in end_times:
                        ready_time = max(ready_time, end_times[prod_id])
            
            if not can_schedule:
                continue
            
            # Check cross-rank recv dependency (must wait for send)
            if node.id in recv_to_send:
                send_id = recv_to_send[node.id]
                if send_id not in scheduled_ids:
                    can_schedule = False
                else:
                    ready_time = max(ready_time, end_times[send_id])
            
            if not can_schedule:
                continue
            
            # Check collective sync (all participants must be ready)
            if node.id in node_to_collective:
                coll = node_to_collective[node.id]
                all_ready = True
                max_ready_time = ready_time
                
                for coll_rank, coll_op in coll.participants:
                    if coll_op.id == node.id:
                        continue
                    # Check if this participant's inputs are ready
                    for inp in coll_op.inputs:
                        if hasattr(inp, 'producer') and inp.producer is not None:
                            prod_id = inp.producer.id
                            if prod_id in node_by_id and prod_id not in scheduled_ids:
                                all_ready = False
                                break
                            if prod_id in end_times:
                                max_ready_time = max(max_ready_time, end_times[prod_id])
                    if not all_ready:
                        break
                
                if not all_ready:
                    continue
                ready_time = max_ready_time
            
            # Schedule this node
            duration = cost_fn(node)
            start_time = ready_time
            end_time = start_time + duration
            
            start_times[node.id] = start_time
            end_times[node.id] = end_time
            scheduled_ids.add(node.id)
            
            scheduled[rank].append(ScheduledOp(
                node=node,
                rank=rank,
                start_us=start_time,
                end_us=end_time,
            ))
            
            rank_current_time[rank] = end_time
            rank_idx[rank] = idx + 1
            made_progress = True
        
        # If a collective was ready, schedule all its participants together
        # (they should all become ready in the same iteration)
    
    return scheduled


# -------- Chrome Trace generation --------

def _make_duration_event(
    scheduled_op: ScheduledOp,
    node_args: Dict | None = None,
) -> Dict:
    """Create a Chrome Trace duration event (ph: X)."""
    node = scheduled_op.node
    kind = _get_display_kind(node)
    
    args = {
        "node_id": node.id,
        "kind": kind,
    }
    
    # Add cost info
    try:
        cost = node.get_cost_meta()
        if cost["flops"] > 0:
            args["flops"] = _format_num(cost["flops"])
        if cost["mem_read"] > 0 or cost["mem_write"] > 0:
            args["mem_read"] = _format_num(cost["mem_read"])
            args["mem_write"] = _format_num(cost["mem_write"])
    except Exception:
        pass
    
    # Add comm-specific info
    if hasattr(node, 'bytes'):
        args["bytes"] = _format_num(node.bytes)
    if hasattr(node, 'group'):
        args["group"] = node.group.match_key
    if hasattr(node, 'dst'):
        args["dst"] = node.dst
    if hasattr(node, 'src') and node.src >= 0:
        args["src"] = node.src
    if hasattr(node, 'label') and node.label:
        args["label"] = node.label
    
    if node_args:
        args.update(node_args)
    
    # Determine category
    if isinstance(node, CommOp):
        cat = "comm"
    elif isinstance(node, ComputeOp):
        cat = "compute"
    else:
        cat = "other"
    
    # Get display name
    name = kind
    if hasattr(node, 'label') and node.label:
        name = f"{kind}: {node.label}"
    
    # Find tensor name from outputs
    for out in node.outputs:
        if hasattr(out, 'name') and out.name:
            name = f"{kind}: {out.name}"
            break
    
    # Check inputs for name
    for inp in node.inputs:
        if hasattr(inp, 'name') and inp.name:
            name = f"{kind}({inp.name})"
            break
    
    return {
        "name": name,
        "cat": cat,
        "ph": "X",
        "ts": scheduled_op.start_us,
        "dur": scheduled_op.end_us - scheduled_op.start_us,
        "pid": scheduled_op.rank,
        "tid": 0,
        "args": args,
    }


def _make_flow_events(
    send_op: ScheduledOp,
    recv_op: ScheduledOp,
    flow_id: int,
) -> List[Dict]:
    """Create flow start/finish events for send→recv dependency."""
    return [
        {
            "name": "p2p",
            "cat": "flow",
            "ph": "s",  # flow start
            "ts": send_op.end_us,  # Arrow leaves at end of send
            "pid": send_op.rank,
            "tid": 0,
            "id": flow_id,
        },
        {
            "name": "p2p",
            "cat": "flow",
            "ph": "f",  # flow finish
            "ts": recv_op.start_us,  # Arrow arrives at start of recv
            "pid": recv_op.rank,
            "tid": 0,
            "id": flow_id,
            "bp": "e",  # bind to enclosing event
        },
    ]


def _make_metadata_events(ranks: List[int]) -> List[Dict]:
    """Create process name metadata events."""
    events = []
    for rank in ranks:
        events.append({
            "name": "process_name",
            "ph": "M",
            "pid": rank,
            "args": {"name": f"GPU {rank}"},
        })
        events.append({
            "name": "thread_name",
            "ph": "M",
            "pid": rank,
            "tid": 0,
            "args": {"name": "Main"},
        })
    return events


def to_chrome_trace(
    multi: MultiRankGraph,
    output: str | Path | None = None,
    cost_fn: Callable[[OpNode], float] = default_cost_fn,
    include_flows: bool = True,
) -> str:
    """Export MultiRankGraph to Chrome Trace JSON format.
    
    Args:
        multi: MultiRankGraph with aggregated graphs
        output: Output file path. If None, returns JSON string.
        cost_fn: Function mapping OpNode to duration in microseconds.
        include_flows: If True, include flow events for send→recv arrows.
    
    Returns:
        JSON string if output is None, otherwise None (writes to file).
    """
    # Schedule operations
    scheduled = schedule_multi_rank(multi, cost_fn=cost_fn)
    
    # Build node.id to ScheduledOp mapping
    scheduled_by_id: Dict[int, ScheduledOp] = {}
    for rank, ops in scheduled.items():
        for sop in ops:
            scheduled_by_id[sop.node.id] = sop
    
    events: List[Dict] = []
    
    # Add metadata events (process/thread names)
    events.extend(_make_metadata_events(multi.ranks))
    
    # Add duration events for all operations (skip scheduling-only ops)
    for rank, ops in scheduled.items():
        for sop in ops:
            # Skip scheduling-only ops (sink, wait_for, completion, etc.)
            # They exist for dependency resolution but don't represent actual kernels
            if is_scheduling_op(sop.node):
                continue
            events.append(_make_duration_event(sop))
    
    # Add flow events for send→recv pairs
    if include_flows:
        flow_id = 0
        for send_op, recv_op, src_rank, dst_rank in multi.send_recv_pairs:
            if send_op.id in scheduled_by_id and recv_op.id in scheduled_by_id:
                send_sop = scheduled_by_id[send_op.id]
                recv_sop = scheduled_by_id[recv_op.id]
                events.extend(_make_flow_events(send_sop, recv_sop, flow_id))
                flow_id += 1
    
    # Build trace object
    trace = {"traceEvents": events}
    
    json_str = json.dumps(trace, indent=2)
    
    if output is not None:
        with open(output, "w") as f:
            f.write(json_str)
        return None
    
    return json_str


def to_chrome_trace_simple(
    multi: MultiRankGraph,
    output: str | Path | None = None,
    spacing_us: float = 100.0,
) -> str:
    """Export using simple topological ordering (no timing simulation).
    
    Each operation gets a fixed duration, ordered by topological position.
    Useful for visualizing graph structure without accurate timing.
    
    Args:
        multi: MultiRankGraph with aggregated graphs
        output: Output file path. If None, returns JSON string.
        spacing_us: Duration per operation in microseconds.
    
    Returns:
        JSON string if output is None, otherwise None (writes to file).
    """
    return to_chrome_trace(
        multi,
        output=output,
        cost_fn=lambda _: spacing_us,
        include_flows=True,
    )
