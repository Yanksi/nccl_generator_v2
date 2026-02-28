from __future__ import annotations

from typing import Dict, List, Optional, Type

from nccl_comm import CommOp as NCCLCommOp
from nccl_primitives import GpuId

from goal import GoalCalc, GoalGraph, GoalGraphNode, GoalOp
from simple_sim.extract import ExtractedGraph, topo_sort
from simple_sim.ir import CommOp as SimCommOp, ComputeOp as SimComputeOp, OpNode, Tensor, Token


def get_comm_op(sim_node: SimCommOp) -> Optional[NCCLCommOp]:
    """Translate a simple_sim CommOp into an nccl_comm CommOp.

    Returns ``None`` when the translation is not yet implemented – the
    caller falls back to a zero-cost ``GoalCalc`` placeholder.

    TODO: implement per-collective-type mapping once the nccl_comm
    parameter construction is figured out.
    """
    return None


def get_cost(cost_meta: dict) -> int:
    """Estimate wall-clock cost (in ns) of a compute op via roofline model.

    TODO: implement a proper roofline analysis that considers hardware
    peak FLOPS and memory bandwidth.  For now, just returns the FLOP
    count as a placeholder duration.
    """
    return cost_meta["flops"]


def simple_ir2goal_ir(
    simple_graph: ExtractedGraph,
    *,
    self_rank: int = 0,
    cpu: int = 0,
    gpu_id2goal_rank: Optional[Dict[GpuId, int]] = None,
    nic: int = 0,
) -> GoalGraph:
    """Translate a simple_sim :class:`ExtractedGraph` into a :class:`GoalGraph`.

    Compute nodes become ``GoalCalc(get_cost(...))``.
    Communication nodes are translated via ``get_comm_op`` → NCCL
    primitives → Goal IR.  When ``get_comm_op`` returns ``None``
    (not yet implemented), a zero-cost ``GoalCalc`` placeholder is
    emitted instead.

    Parameters
    ----------
    simple_graph : ExtractedGraph
        The computation graph produced by ``simple_sim.extract.get_graph``.
    self_rank : int
        Goal rank for the generated ops.
    cpu : int
        CPU id for the generated ops.
    gpu_id2goal_rank : dict, optional
        Mapping from ``GpuId`` to goal rank, required for actual NCCL
        comm translation.  May be ``None`` while ``get_comm_op`` is
        still a stub.
    nic : int
        NIC id for communication ops.
    """
    sorted_nodes = topo_sort(simple_graph.nodes)

    # Map simple_sim node id → GoalGraphNode
    node_map: Dict[int, GoalGraphNode] = {}
    graph_nodes: List[GoalGraphNode] = []

    for node in sorted_nodes:
        # --- collect predecessor GoalGraphNodes -----------------------
        predecessors = _get_predecessors(node, node_map)

        # --- translate to GoalOp -------------------------------------
        if isinstance(node, SimCommOp):
            goal_op = _translate_comm(
                node,
                self_rank=self_rank,
                cpu=cpu,
                gpu_id2goal_rank=gpu_id2goal_rank or {},
                nic=nic,
            )
        else:
            # ComputeOp (including scheduling-only ops with zero cost)
            cost = get_cost(node.get_cost_meta())
            goal_op = GoalCalc(cost, self_rank=self_rank, cpu=cpu)

        gnode = GoalGraphNode(goal_op, predecessors)
        node_map[node.id] = gnode
        graph_nodes.append(gnode)

    return GoalGraph(graph_nodes, self_rank=self_rank, cpu=cpu)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_predecessors(
    node: OpNode,
    node_map: Dict[int, GoalGraphNode],
) -> List[GoalGraphNode]:
    """Return deduplicated predecessor ``GoalGraphNode``s for *node*.

    A predecessor is any node in *node_map* that produces a
    ``Tensor`` or ``Token`` consumed by *node*.
    """
    seen: set[int] = set()  # id(GoalGraphNode) for dedup
    preds: List[GoalGraphNode] = []
    for inp in node.inputs:
        producer: Optional[OpNode] = None
        if isinstance(inp, Tensor):
            producer = inp.producer
        elif isinstance(inp, Token):
            producer = inp.producer
        if producer is not None and producer.id in node_map:
            gnode = node_map[producer.id]
            if id(gnode) not in seen:
                seen.add(id(gnode))
                preds.append(gnode)
    return preds


def _translate_comm(
    node: SimCommOp,
    *,
    self_rank: int,
    cpu: int,
    gpu_id2goal_rank: Dict[GpuId, int],
    nic: int,
) -> GoalOp:
    """Translate a simple_sim CommOp into a GoalOp.

    Tries ``get_comm_op`` first.  If the factory returns a valid
    ``NCCLCommOp``, converts via ``.to_primitives().to_goal()``.
    Otherwise falls back to a zero-cost ``GoalCalc`` placeholder.
    """
    nccl_op = get_comm_op(node)
    if nccl_op is not None:
        goal_op, _next_cpu = nccl_op.to_goal(gpu_id2goal_rank, cpu, nic)
        return goal_op

    # Fallback: placeholder calc with zero cost.
    return GoalCalc(0, self_rank=self_rank, cpu=cpu)