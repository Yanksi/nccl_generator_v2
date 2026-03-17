from __future__ import annotations

from typing import Dict, List, Optional, Type

from communication import CommOp, Communicator, CollDevice, CollAlgo
from nccl_primitives import GpuId

import pickle
from collections import defaultdict
import pathlib
from functools import reduce
from tqdm import tqdm

from goal import GoalCalc, GoalGraph, GoalGraphNode, GoalOp, GoalRank
from simple_sim.extract import ExtractedGraph, topo_sort
from simple_sim.ir import CommOp as SimCommOp, ComputeOp as SimComputeOp, OpNode, Tensor, Token, Group
from simple_sim.ops_comm import AllReduceOp, AllGatherOp, ReduceScatterOp, SendOp, FillOp as RecvOp

context_dict = {
    "tp": 1,
    "zero1": 2,
    "pp": 3,
}

device2goal_rank = {rank: rank for rank in range(32)}

def get_context(sim_node: SimCommOp) -> int:
    """Map simple_sim comm op context string to an integer for Goal IR."""
    # node_label = sim_node.label if sim_node.label else "unknown"
    # tag = ((hash(node_label) & 0xFFFF) % 1000) * 100 + context_dict.get(sim_node.context, 0)
    # return tag
    return context_dict.get(sim_node.context, 0)

def translate_comm_node(sim_node: SimCommOp, communicators: Dict[str, Communicator], device2goal_rank, *, cpu: int = 0) -> Optional[CommOp]:
    """Translate a simple_sim CommOp into an nccl_comm CommOp.

    Returns ``None`` when the translation is not yet implemented – the
    caller falls back to a zero-cost ``GoalCalc`` placeholder.

    TODO: implement per-collective-type mapping once the nccl_comm
    parameter construction is figured out.
    """
    comm = communicators.get(sim_node.group.match_key)
    if comm is None:
        print(f"Warning: no communicator found for group {sim_node.group}, cannot translate {sim_node}")
        return None
    tensor_size = sim_node.inputs[0].shape
    n_elements = reduce(lambda x, y: x * y, tensor_size, 1) * 2
    context = get_context(sim_node)
    if isinstance(sim_node, AllReduceOp):
        op = comm.allreduce(size=n_elements, context=context, algo=CollAlgo.RECURSIVE_DOUBLING)
    elif isinstance(sim_node, AllGatherOp):
        op = comm.allgather(size=n_elements, context=context, algo=CollAlgo.RING)
    elif isinstance(sim_node, ReduceScatterOp):
        op = comm.reducescatter(size=n_elements, context=context, algo=CollAlgo.RING)
    elif isinstance(sim_node, SendOp):
        op = comm.send(size=n_elements, context=context, dst_rank=sim_node.dst)
    elif isinstance(sim_node, RecvOp):
        op = comm.recv(size=n_elements, context=context, src_rank=sim_node.src)
    else:
        print(f"Warning: unrecognized comm op type {type(sim_node)}, cannot translate {sim_node}")
        return None
    return op.to_goal(device2goal_rank, cpu, nic=0)


# def get_cost(cost_meta: dict) -> int:
#     """Estimate wall-clock cost (in ns) of a compute op via roofline model.

#     TODO: implement a proper roofline analysis that considers hardware
#     peak FLOPS and memory bandwidth.  For now, just returns the FLOP
#     count as a placeholder duration.
#     """
#     return cost_meta["flops"]

def translate_compute_node(compute_node: SimComputeOp, *, cpu: int = 0) -> GoalOp:
    cost = compute_node.get_cost_meta()
    flops = cost.get("flops", 0)
    mem = cost.get("mem_read", 0) + cost.get("mem_write", 0)
    
    # Simple roofline: assume 1 GFLOP/s compute, 1 GB/s memory
    # Time = max(compute_time, memory_time)
    compute_us = flops // 1e6  # 1 TFLOP/s = 1e12 FLOP/s = 1e6 FLOP/µs
    memory_us = mem  // 1e6  # 1 TB/s = 1e12 B/s, assume 2 bytes/element
    
    duration = max(compute_us, memory_us)  # Minimum 1µs
    return GoalCalc(int(duration), cpu=cpu)

def extract_communicators(rank_nodes: Dict[int, List[OpNode]]) -> Dict[str, Communicator]:
    participating_ranks = defaultdict(dict)
    for rank, nodes in rank_nodes.items():
        for node in nodes:
            if isinstance(node, SimCommOp):
                group = node.group
                participating_ranks[group.match_key][group.self_rank] = rank
    for match_key, rank_map in participating_ranks.items():
        participating_ranks[match_key] = [rank_map[i] for i in range(len(rank_map))]
    communicators = {}
    for match_key, rank_map in participating_ranks.items():
        communicators[match_key] = Communicator(rank_map, match_key)
    return communicators


def get_graphs(graphs_dir: pathlib.Path):
    graph_paths = sorted(graphs_dir.glob("*.pkl"))
    graphs = {}
    for rank, path in enumerate(graph_paths):
        with open(path, 'rb') as f:
            graph = pickle.load(f)
            # return the graph as a list of topologically sorted nodes
            graphs[rank] = topo_sort(graph.nodes)
    return graphs
    
def simple_ir2goal_ir(
    nodes: List[OpNode],
    communicators: Dict[str, Communicator],
    *,
    cpu: int = 0,
    nic: int = 0,
) -> GoalGraph:
    """Translate a simple_sim :class:`ExtractedGraph` into a :class:`GoalGraph`.

    Compute nodes become ``GoalCalc(get_cost(...))``.
    Communication nodes are translated via ``get_comm_op`` → Goal IR.
    When ``get_comm_op`` returns ``None`` (not yet implemented),
    a zero-cost ``GoalCalc`` placeholder is emitted instead.
    """

    # Map simple_sim node id → GoalGraphNode
    node_map: Dict[int, GoalGraphNode] = {}
    graph_nodes: List[GoalGraphNode] = []

    for node in nodes:
        # --- collect predecessor GoalGraphNodes -----------------------
        predecessors = _get_predecessors(node, node_map)

        # --- translate to GoalOp -------------------------------------
        if isinstance(node, SimCommOp):
            goal_op = translate_comm_node(node, communicators, device2goal_rank, cpu=cpu)
        else:
            goal_op = translate_compute_node(node, cpu=cpu)

        gnode = GoalGraphNode(goal_op, predecessors)
        node_map[node.id] = gnode
        graph_nodes.append(gnode)

    return GoalGraph(graph_nodes, cpu=cpu)


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

if __name__ == "__main__":
    graphs = get_graphs(pathlib.Path("llama3_graphs"))
    goal_path = pathlib.Path("llama3_goal")
    goal_path.mkdir(exist_ok=True)
    communicators = extract_communicators(graphs)
    # print(f"Extracted communicators: {communicators}")
    with open("llama3.goal", "w") as f:
        f.write(f"num_ranks {len(graphs)}\n")
        for rank, nodes in tqdm(graphs.items(), desc="Translating to Goal IR"):
            with CollDevice(rank):
                goal_graph = simple_ir2goal_ir(nodes, communicators, cpu=0, nic=0)
                f.write(f"rank {rank} {{\n")
                for line in goal_graph.generate_lines():
                    f.write(f"{line}\n")
                f.write("}\n")
    # for rank, nodes in graphs.items():
    #     with CollDevice(rank) as coll_device:
    #         goal_graph = simple_ir2goal_ir(nodes, communicators, cpu=0, nic=0)
    #         print(f"Goal graph for rank {rank} with {len(goal_graph.nodes)} nodes:")
    #         with open(goal_path / f"{rank:02d}.goal", "w") as f:
    #             for line in goal_graph.generate_lines():
    #                 f.write(line + "\n")
    # sample_graph_path = "graphs/llama3_training_device0.pkl"
    # with open(sample_graph_path, 'rb') as f:
    #     simple_graph = pickle.load(f)
    #     print(f"Loaded simple graph from {sample_graph_path} with {len(simple_graph.nodes)} nodes")
    # goal_graph = simple_ir2goal_ir(simple_graph, self_rank=0, cpu=0)
    # print(goal_graph)