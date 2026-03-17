"""Multi-rank graph aggregation for cross-GPU visualization.

This module provides infrastructure to aggregate per-rank computation graphs
into a unified multi-rank view, matching send/recv pairs and grouping
collective operations across ranks.

Usage:
    from simple_sim import build_rank, get_graph
    from simple_sim.aggregate import aggregate_graphs, MultiRankGraph
    
    # Build per-rank graphs
    graphs = {}
    for rank in range(4):
        with build_rank(rank):
            g = build_model(...)
            graphs[rank] = g
    
    # Aggregate into multi-rank view
    multi_graph = aggregate_graphs(graphs)
    
    # Access matched send/recv pairs and collective groups
    for send_op, recv_op, src, dst in multi_graph.send_recv_pairs:
        print(f"Send {src} -> Recv {dst}")
"""
from __future__ import annotations

import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, TYPE_CHECKING

from .extract import ExtractedGraph, topo_sort
from .ir import CommOp, Group, OpNode

if TYPE_CHECKING:
    from .ops_comm import SendOp, FillOp, AllReduceOp, ReduceScatterOp, AllGatherOp


# -------- Data structures --------

@dataclass
class CrossRankCollective:
    """Virtual node representing a collective operation across multiple ranks.
    
    This groups the per-rank collective ops that correspond to the same
    logical collective (e.g., one AllReduce across 4 GPUs).
    
    Attributes:
        kind: Collective type ("allreduce", "reduce_scatter", "allgather")
        group_key: The Group.match_key identifying this communicator
        group: The Group object (from first participant)
        participants: List of (rank, per-rank CommOp) tuples
        call_index: Index of this collective among same (group_key, kind) calls
    """
    kind: str
    group_key: str
    group: Group
    participants: List[Tuple[int, CommOp]]
    call_index: int = 0


@dataclass
class MultiRankGraph:
    """Aggregated computation graph across multiple GPU ranks.
    
    Attributes:
        graphs: Per-rank ExtractedGraph instances keyed by rank ID
        send_recv_pairs: Matched (SendOp, FillOp, src_rank, dst_rank) tuples
        cross_collectives: List of CrossRankCollective grouping collective ops
    """
    graphs: Dict[int, ExtractedGraph]
    send_recv_pairs: List[Tuple["SendOp", "FillOp", int, int]] = field(default_factory=list)
    cross_collectives: List[CrossRankCollective] = field(default_factory=list)
    
    @property
    def ranks(self) -> List[int]:
        """Return sorted list of rank IDs."""
        return sorted(self.graphs.keys())
    
    @property
    def num_ranks(self) -> int:
        """Return number of ranks."""
        return len(self.graphs)
    
    def all_nodes(self) -> List[Tuple[int, OpNode]]:
        """Return all nodes across all ranks as (rank, node) tuples."""
        result = []
        for rank, g in self.graphs.items():
            for node in g.nodes:
                result.append((rank, node))
        return result
    
    def save(self, path: str | Path) -> None:
        """Save the multi-rank graph to a pickle file."""
        with open(path, "wb") as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str | Path) -> "MultiRankGraph":
        """Load a multi-rank graph from a pickle file."""
        with open(path, "rb") as f:
            return pickle.load(f)


# -------- Matching functions --------

def _match_send_recv(
    graphs: Dict[int, ExtractedGraph]
) -> List[Tuple["SendOp", "FillOp", int, int]]:
    """Match SendOp to FillOp across ranks.
    
    Matching logic:
    - SendOp on rank S with dst=D and label=L matches
    - FillOp on rank D with src=S and label=L (recv mode, not local fill)
    
    Returns:
        List of (send_op, fill_op, src_rank, dst_rank) tuples
    """
    from .ops_comm import SendOp, FillOp
    
    # Collect sends: key = (dst_rank, label), value = (src_rank, SendOp)
    sends: Dict[Tuple[int, str], Tuple[int, "SendOp"]] = {}
    
    for rank, g in graphs.items():
        for node in g.nodes:
            if isinstance(node, SendOp):
                key = (node.dst, node.label)
                sends[key] = (rank, node)
    
    # Match recvs (FillOp in recv mode: src >= 0)
    matches = []
    for rank, g in graphs.items():
        for node in g.nodes:
            if isinstance(node, FillOp) and node.src >= 0:
                # This is a recv operation
                # Key: this rank is the destination, node.src is the source
                key = (rank, node.label)
                if key in sends:
                    src_rank, send_op = sends[key]
                    # Verify the source matches
                    if src_rank == node.src:
                        matches.append((send_op, node, src_rank, rank))
    
    return matches


def _group_collectives(
    graphs: Dict[int, ExtractedGraph]
) -> List[CrossRankCollective]:
    """Group collective operations across ranks.

    Primary matching: by ``(group.match_key, kind, label)``.
    Fallback: if multiple ops share the same ``(group_key, kind, label)`` on
    any single rank (i.e. the label is not unique within that bucket), they are
    disambiguated by topological order within the bucket — the N-th op on each
    rank is matched with the N-th op on every other rank.

    Returns:
        List of CrossRankCollective objects
    """
    from .ops_comm import AllReduceOp, ReduceScatterOp, AllGatherOp

    collective_types = (AllReduceOp, ReduceScatterOp, AllGatherOp)

    # Step 1: For each rank, bucket collectives by (group_key, kind, label),
    # preserving topological order within each bucket for disambiguation.
    rank_label_buckets: Dict[int, Dict[Tuple[str, str, str], List[CommOp]]] = {}

    for rank, g in graphs.items():
        topo_nodes = topo_sort(g.nodes)
        buckets: Dict[Tuple[str, str, str], List[CommOp]] = defaultdict(list)
        for node in topo_nodes:
            if isinstance(node, collective_types):
                key = (node.group.match_key, node.kind, node.label)
                buckets[key].append(node)
        rank_label_buckets[rank] = dict(buckets)

    # Step 2: Collect all unique (group_key, kind, label) buckets.
    all_label_keys: set[Tuple[str, str, str]] = set()
    for buckets in rank_label_buckets.values():
        all_label_keys.update(buckets.keys())

    # Step 3: For each bucket, emit one CrossRankCollective per call index.
    # When the bucket has only one op per rank (the common case), call_index=0
    # and there is no ambiguity.  When a bucket has N>1 ops per rank, the
    # N collectives are matched positionally (topo order fallback).
    cross_collectives = []

    for group_key, kind, label in sorted(all_label_keys):
        max_calls = max(
            len(rank_label_buckets.get(r, {}).get((group_key, kind, label), []))
            for r in graphs.keys()
        )

        for call_idx in range(max_calls):
            participants = []
            group = None

            for rank in sorted(graphs.keys()):
                ops = rank_label_buckets.get(rank, {}).get((group_key, kind, label), [])
                if call_idx < len(ops):
                    op = ops[call_idx]
                    participants.append((rank, op))
                    if group is None:
                        group = op.group

            if len(participants) > 1 and group is not None:
                cross_collectives.append(CrossRankCollective(
                    kind=kind,
                    group_key=group_key,
                    group=group,
                    participants=participants,
                    call_index=call_idx,
                ))

    return cross_collectives


# -------- Main aggregation function --------

def aggregate_graphs(
    graphs: Dict[int, ExtractedGraph],
    merge_cross_rank: bool = False,
) -> MultiRankGraph:
    """Aggregate per-rank graphs into a multi-rank view.
    
    This function:
    1. Validates that all nodes have rank information
    2. Matches send/recv pairs across ranks by (src, dst, label)
    3. Groups collectives by (group.match_key, kind, label); falls back to
       topological order within a bucket when the label is non-unique on a rank
    4. Optionally merges cross-rank operations into unified nodes
    
    Args:
        graphs: Dictionary mapping rank ID to ExtractedGraph
        merge_cross_rank: If True, actually merge cross-rank operations:
            - Send/recv pairs: recv takes send's output as a dependency
            - Collectives: replaced with a single merged node spanning all ranks
    
    Returns:
        MultiRankGraph with matched send/recv pairs and collective groups
    """
    # Validate rank information
    for rank, g in graphs.items():
        for node in g.nodes:
            if node.rank is not None and node.rank != rank:
                raise ValueError(
                    f"Node {node.id} has rank={node.rank} but is in graph for rank={rank}"
                )
    
    # Match send/recv pairs
    send_recv_pairs = _match_send_recv(graphs)
    
    # Group collectives
    cross_collectives = _group_collectives(graphs)
    
    multi = MultiRankGraph(
        graphs=graphs,
        send_recv_pairs=send_recv_pairs,
        cross_collectives=cross_collectives,
    )
    
    # Apply merge if requested
    if merge_cross_rank:
        _apply_cross_rank_merge(multi)
    
    return multi


def _apply_cross_rank_merge(multi: MultiRankGraph) -> None:
    """Merge cross-rank operations into unified nodes.
    
    This modifies the graphs in-place:
    - Send/recv: recv gets send's output as a dependency input
    - Collectives: replaced with MergedCollectiveOp spanning all ranks
    """
    _merge_send_recv_pairs(multi)
    _merge_collective_ops(multi)


def _merge_send_recv_pairs(multi: MultiRankGraph) -> None:
    """Merge send/recv pairs: recv takes send's output as dependency.
    
    This creates a proper data-flow edge from send to recv across ranks.
    """
    from .ir import Tensor, Token
    
    for send_op, recv_op, src_rank, dst_rank in multi.send_recv_pairs:
        # Find the tensor produced by send_op by scanning graph
        # (send_op.outputs may be empty since outputs aren't always populated)
        sent_tensor = None
        
        # First try send_op.outputs
        if send_op.outputs:
            sent_tensor = send_op.outputs[0]
        else:
            # Scan the source rank's graph for tensors produced by send_op
            src_graph = multi.graphs[src_rank]
            for node in src_graph.nodes:
                for inp in node.inputs:
                    if hasattr(inp, 'producer') and inp.producer is send_op:
                        sent_tensor = inp
                        break
                if sent_tensor:
                    break
            
            # Also check if send's input tensor itself should be used as dependency
            # (since send produces an alias of its input)
            if sent_tensor is None and send_op.inputs:
                # Use the send's input as the dependency marker
                # This creates a path: send_input -> send_op -> recv_op
                sent_tensor = send_op.inputs[0]
        
        if sent_tensor is not None:
            # Check if already connected (idempotent)
            if sent_tensor not in recv_op.inputs:
                # Use object.__setattr__ to bypass frozen constraint
                new_inputs = tuple(recv_op.inputs) + (sent_tensor,)
                object.__setattr__(recv_op, "inputs", new_inputs)


def _merge_collective_ops(multi: MultiRankGraph) -> None:
    """Merge collective operations into single nodes spanning all ranks.
    
    For each collective group:
    - Creates a MergedCollectiveOp that replaces all per-rank collective ops
    - The merged op takes all per-rank inputs and produces all per-rank outputs
    - Per-rank ops are removed from their graphs
    """
    from .ops_comm import AllReduceOp, ReduceScatterOp, AllGatherOp
    from .ir import Tensor, Token
    
    # Build a mapping from op to its output tensors by scanning all tensors
    op_to_outputs: Dict[int, List] = {}  # op.id -> list of output tensors
    for rank, g in multi.graphs.items():
        for node in g.nodes:
            for inp in node.inputs:
                if hasattr(inp, 'producer') and inp.producer is not None:
                    pid = inp.producer.id
                    if pid not in op_to_outputs:
                        op_to_outputs[pid] = []
                    if inp not in op_to_outputs[pid]:
                        op_to_outputs[pid].append(inp)
    
    for coll in multi.cross_collectives:
        if len(coll.participants) <= 1:
            continue
        
        # Collect all inputs and outputs from participating ops
        all_inputs = []
        all_outputs = []
        participant_ops = []
        
        for rank, op in sorted(coll.participants, key=lambda x: x[0]):
            all_inputs.extend(op.inputs)
            # Get outputs by looking at tensors that have this op as producer
            if op.id in op_to_outputs:
                all_outputs.extend(op_to_outputs[op.id])
            participant_ops.append((rank, op))
        
        # Create merged collective node with all inputs and outputs
        first_op = coll.participants[0][1]
        merged_op = MergedCollectiveOp(
            kind=coll.kind,
            group=coll.group,
            label=first_op.label,
            context=first_op.context,
            participants=tuple(participant_ops),
            inputs=tuple(all_inputs),
            outputs=tuple(all_outputs),
        )
        
        # Update outputs to point to merged op (for proper edge traversal)
        for out in all_outputs:
            # Use object.__setattr__ to bypass frozen constraint on Tensor
            if hasattr(out, 'producer'):
                object.__setattr__(out, 'producer', merged_op)
        
        # Remove original ops from their graphs and insert merged op once
        first_rank = True
        for rank, op in participant_ops:
            g = multi.graphs[rank]
            if op in g.nodes:
                g.nodes.remove(op)
            
            # Add merged op to the first rank's graph only
            if first_rank:
                g.nodes.append(merged_op)
                first_rank = False
        
        # Store merged op in the collective for later reference
        coll.merged_op = merged_op


@dataclass(frozen=True, eq=False)
class MergedCollectiveOp(CommOp):
    """A collective operation merged across multiple ranks.
    
    This represents a single logical collective (e.g., AllReduce) that
    spans multiple GPU ranks. It takes inputs from all ranks and produces
    outputs for all ranks.
    
    Attributes:
        kind: "allreduce", "reduce_scatter", "allgather"
        group: The communicator group
        label: Inherited from the per-rank ops; not needed for matching post-aggregation
        participants: Tuple of (rank, original_op) pairs
        inputs: All input tensors from all ranks
        outputs: All output tensors going to all ranks
    """
    kind: str = "merged_collective"
    label: str = ""  # override: optional for merged ops (matching not needed post-aggregation)
    participants: Tuple[Tuple[int, CommOp], ...] = field(default_factory=tuple)
    
    @property
    def display_kind(self) -> str:
        return f"merged_{self.kind}"
    
    @property
    def bytes(self) -> int:
        from .ir import Tensor
        from .utils import bytes_of
        total = 0
        for inp in self.inputs:
            if isinstance(inp, Tensor):
                total += bytes_of(inp.physical_shape(), inp.dtype)
        return total
    
    @property
    def participating_ranks(self) -> List[int]:
        return sorted(r for r, _ in self.participants)


# Update CrossRankCollective to hold optional merged_op
CrossRankCollective.merged_op = None


def save_graph(graph: ExtractedGraph, path: str | Path) -> None:
    """Save a single-rank graph to a pickle file."""
    with open(path, "wb") as f:
        pickle.dump(graph, f)


def load_graph(path: str | Path) -> ExtractedGraph:
    """Load a single-rank graph from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)
