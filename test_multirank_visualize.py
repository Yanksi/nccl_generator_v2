"""Test script for multi-rank graph aggregation and Chrome Trace visualization.

This script demonstrates:
1. Building computation graphs for multiple GPU ranks
2. Aggregating them with send/recv matching
3. Exporting to Chrome Trace JSON (viewable in chrome://tracing)
4. Exporting individual rank graphs to SVG

Usage:
    uv run python test_multirank_visualize.py

Output:
    - multirank_timeline.json  (open in chrome://tracing, enable Flow Events)
    - multirank_rank0.svg      (per-rank graph visualization)
    - multirank_rank1.svg
"""

from simple_sim import (
    # Core types
    Tensor, Group, ShardSpec, MemoryCategory,
    # Constructors
    input_tensor, parameter, matmul, add, activation, detach,
    # Communication
    fill, send, allreduce, allgather, reduce_scatter, sink,
    # Graph extraction
    get_graph, backward,
    # Multi-rank support
    build_rank, aggregate_graphs,
    # Visualization
    visualize_graph, visualize_multi_rank_graph, to_chrome_trace,
    # Optimizer
    Zero1Plan, zero1_optimizer_step,
    # TP primitives
    megatron_mlp,
)


def build_rank_0_graph(dp_group: Group) -> tuple:
    """Build computation graph for rank 0.
    
    Rank 0 does:
    1. Input + MatMul
    2. Send activation to rank 1
    3. AllReduce (syncs with rank 1)
    """
    x = input_tensor((4, 8), name="x")
    w1 = parameter((8, 16), name="w1")
    
    # Forward: matmul
    h = matmul(x, w1, name="h")
    
    # Send to rank 1
    pp_group = Group("pp", size=2, name="pp_world")
    sent = send(h, dst=1, group=pp_group, label="fwd_act", name="send_h")
    
    # AllReduce (e.g., gradient sync)
    ar_out, ar_tok = allreduce(h, group=dp_group, label="allreduce_h", name="allreduce_h")
    
    # Get graph rooted at send and allreduce outputs
    g = get_graph(sent, ar_out)
    return g


def build_rank_1_graph(dp_group: Group) -> tuple:
    """Build computation graph for rank 1.
    
    Rank 1 does:
    1. Recv activation from rank 0
    2. MatMul on received data
    3. AllReduce (syncs with rank 0)
    """
    # Placeholder for recv
    placeholder = Tensor(
        shape=(4, 16),
        dtype="fp16",
        memory_category=MemoryCategory.NOT_MATERIALIZED,
        requires_grad=False,
        name="recv_placeholder",
    )
    
    # Recv from rank 0
    pp_group = Group("pp", size=2, name="pp_world")
    h = fill(placeholder, src=0, group=pp_group, label="fwd_act", name="recv_h")
    
    # Forward: matmul
    w2 = parameter((16, 4), name="w2")
    out = matmul(h, w2, name="out")
    
    # AllReduce (e.g., gradient sync)
    ar_out, ar_tok = allreduce(out, group=dp_group, label="allreduce_out", name="allreduce_out")
    
    # Get graph
    g = get_graph(out, ar_out)
    return g


# ===========================================================================
# 4-GPU Example: TP=2, DP=2, no PP
# ===========================================================================
# Layout:
#   Rank 0: TP group 0, DP group 0
#   Rank 1: TP group 0, DP group 1
#   Rank 2: TP group 1, DP group 0
#   Rank 3: TP group 1, DP group 1
#
# TP groups (for AllReduce in MLP):
#   TP0: ranks 0, 1
#   TP1: ranks 2, 3
#
# DP groups (for ZeRO-1 gradient sync):
#   DP0: ranks 0, 2
#   DP1: ranks 1, 3
# ===========================================================================

def build_tp_dp_graph(
    rank: int,
    tp_group: Group,
    dp_group: Group,
    hidden: int = 256,
    inter: int = 512,
    batch: int = 8,
) -> "Graph":
    """
    Build computation graph for one GPU in a TP=2, DP=2 setup.
    
    Each rank does:
    Training:
      1. Input (replicated across TP group)
      2. Column-parallel matmul (w1 sharded on output dim)
      3. Activation
      4. Row-parallel matmul (w2 sharded on input dim) 
      5. AllReduce across TP group (sum partial outputs)
      6. Backward pass
      7. ZeRO-1: ReduceScatter gradients across DP group
      8. Adam update on shard
      9. AllGather parameters across DP group
    
    Inference:
      1. Forward pass on updated weights (no grad)
    """
    tp_size = tp_group.size
    
    # ---- Parameters (TP-sharded) ----
    # Column-parallel: sharded on output features (axis=1)
    w1_col = parameter(
        (hidden, inter),
        name="w1_col",
        tp_group=tp_group,
        shard=ShardSpec("sharded", axis=1, parts=tp_size),
    )
    # Row-parallel: sharded on input features (axis=0)
    w2_row = parameter(
        (inter, hidden),
        name="w2_row",
        tp_group=tp_group,
        shard=ShardSpec("sharded", axis=0, parts=tp_size),
    )
    params = [w1_col, w2_row]
    
    # ---- Training: Forward Pass ----
    # Input is replicated across TP group
    x = input_tensor((batch, hidden), name="train.x")
    
    # Megatron-style MLP: column-parallel -> activation -> row-parallel -> AllReduce
    y = megatron_mlp(x, w1_col, w2_row, tp_group=tp_group, name="train.mlp")
    
    # ---- Training: Backward Pass ----
    grads = backward(y, wrt=params)
    
    # ---- Training: ZeRO-1 Optimizer ----
    plan = Zero1Plan(dp_group=dp_group, gather_policy="eager_allgather")
    new_params, opt_done = zero1_optimizer_step(params, grads, plan=plan, name="zero1")
    
    # ---- Inference: Forward Pass (no grad) ----
    # Detach params for inference
    inf_w1 = detach(new_params[0], name="inf.w1_col")
    inf_w2 = detach(new_params[1], name="inf.w2_row")
    
    inf_x = input_tensor((batch, hidden), name="inf.x")
    
    # Megatron-style MLP for inference
    inf_y = megatron_mlp(inf_x, inf_w1, inf_w2, tp_group=tp_group, name="inf.mlp")
    
    # Collect all outputs
    done = sink(opt_done, inf_y, name="done")
    g = get_graph(done)
    return g


def main():
    print("=" * 60)
    print("Multi-Rank Graph Visualization Test")
    print("=" * 60)
    
    # Define shared communicator group
    dp_group = Group("dp", size=2, name="dp_world")
    
    # Build per-rank graphs
    graphs = {}
    
    print("\n[1] Building rank 0 graph...")
    with build_rank(0):
        graphs[0] = build_rank_0_graph(dp_group)
        print(f"    Nodes: {len(graphs[0].nodes)}")
        for n in graphs[0].nodes:
            print(f"      {n.kind} (id={n.id}, rank={n.rank})")
    
    print("\n[2] Building rank 1 graph...")
    with build_rank(1):
        graphs[1] = build_rank_1_graph(dp_group)
        print(f"    Nodes: {len(graphs[1].nodes)}")
        for n in graphs[1].nodes:
            print(f"      {n.kind} (id={n.id}, rank={n.rank})")
    
    # Aggregate graphs (without merging)
    print("\n[3] Aggregating graphs (non-merged mode)...")
    multi = aggregate_graphs(graphs, merge_cross_rank=False)
    print(f"    Ranks: {multi.ranks}")
    print(f"    Send/Recv pairs: {len(multi.send_recv_pairs)}")
    for send_op, recv_op, src, dst in multi.send_recv_pairs:
        print(f"      Send(rank={src}, label={send_op.label!r}) -> Recv(rank={dst}, label={recv_op.label!r})")
    print(f"    Cross-rank collectives: {len(multi.cross_collectives)}")
    for coll in multi.cross_collectives:
        ranks = [r for r, _ in coll.participants]
        print(f"      {coll.kind} on {coll.group_key}: ranks {ranks}")
    
    # Export per-rank SVG graphs
    print("\n[4] Generating per-rank SVG graphs...")
    for rank, g in graphs.items():
        output_path = f"multirank_rank{rank}"
        visualize_graph(g, output=output_path, format="svg", show_tensors=False)
        print(f"    Saved: {output_path}.svg")
    
    # Export aggregated multi-rank graph (non-merged mode)
    print("\n[5] Generating aggregated multi-rank SVG graph (non-merged)...")
    output_agg = "multirank_aggregated"
    visualize_multi_rank_graph(
        multi,
        output=output_agg,
        format="svg",
        title="Multi-GPU Computation Graph (Non-Merged)",
        show_costs=True,
        show_tensors=True,
        merged_mode=False,
    )
    print(f"    Saved: {output_agg}.svg")
    
    # Now create a merged version
    print("\n[6] Aggregating graphs (merged mode)...")
    # Need fresh graphs since merge modifies in-place
    graphs_for_merge = {}
    with build_rank(0):
        graphs_for_merge[0] = build_rank_0_graph(dp_group)
    with build_rank(1):
        graphs_for_merge[1] = build_rank_1_graph(dp_group)
    
    multi_merged = aggregate_graphs(graphs_for_merge, merge_cross_rank=True)
    print(f"    Send/recv pairs merged (recv now depends on send)")
    print(f"    Collectives merged into single nodes")
    
    # Export merged aggregated graph
    print("\n[7] Generating aggregated multi-rank SVG graph (merged)...")
    output_merged = "multirank_merged"
    visualize_multi_rank_graph(
        multi_merged,
        output=output_merged,
        format="svg",
        title="Multi-GPU Computation Graph (Merged)",
        show_costs=True,
        show_tensors=True,
        merged_mode=True,
    )
    print(f"    Saved: {output_merged}.svg")
    
    # Export Chrome Trace JSON
    print("\n[8] Generating Chrome Trace JSON...")
    output_json = "multirank_timeline.json"
    to_chrome_trace(multi, output=output_json)
    print(f"    Saved: {output_json}")
    print(f"    Open in: chrome://tracing or https://ui.perfetto.dev/")
    print(f"    Note: Enable 'Flow Events' button to see send->recv arrows")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


def main_4gpu():
    """
    4-GPU example: TP=2, DP=2, one training iteration + inference.
    
    Layout:
      Rank 0: TP group 0, DP group 0
      Rank 1: TP group 0, DP group 1  
      Rank 2: TP group 1, DP group 0
      Rank 3: TP group 1, DP group 1
    
    TP groups (for AllReduce in MLP):
      TP0: ranks 0, 1
      TP1: ranks 2, 3
    
    DP groups (for ZeRO-1 gradient sync):
      DP0: ranks 0, 2
      DP1: ranks 1, 3
    """
    print("\n" + "=" * 60)
    print("4-GPU Example: TP=2, DP=2, Training + Inference")
    print("=" * 60)
    
    # Each rank sees its own TP and DP group handles
    # TP groups: ranks 0,1 are in TP0; ranks 2,3 are in TP1
    # DP groups: ranks 0,2 are in DP0; ranks 1,3 are in DP1
    
    def get_tp_group(rank: int) -> Group:
        """Get TP group for this rank."""
        tp_id = rank // 2  # 0,1 -> 0; 2,3 -> 1
        return Group("tp", size=2, name=f"tp{tp_id}")
    
    def get_dp_group(rank: int) -> Group:
        """Get DP group for this rank."""
        dp_id = rank % 2  # 0,2 -> 0; 1,3 -> 1
        return Group("dp", size=2, name=f"dp{dp_id}")
    
    # Build graphs for all 4 ranks
    graphs = {}
    for rank in range(4):
        print(f"\n[{rank+1}] Building rank {rank} graph...")
        tp_group = get_tp_group(rank)
        dp_group = get_dp_group(rank)
        print(f"    TP group: {tp_group.name}, DP group: {dp_group.name}")
        
        with build_rank(rank):
            graphs[rank] = build_tp_dp_graph(
                rank=rank,
                tp_group=tp_group,
                dp_group=dp_group,
                hidden=256,
                inter=512,
                batch=8,
            )
            print(f"    Nodes: {len(graphs[rank].nodes)}")
    
    # Aggregate graphs (without merging)
    print("\n[5] Aggregating 4-GPU graphs (non-merged mode)...")
    multi = aggregate_graphs(graphs, merge_cross_rank=False)
    print(f"    Ranks: {multi.ranks}")
    print(f"    Send/Recv pairs: {len(multi.send_recv_pairs)}")
    print(f"    Cross-rank collectives: {len(multi.cross_collectives)}")
    
    # Categorize collectives by type
    tp_collectives = [c for c in multi.cross_collectives if "tp" in c.group_key.lower()]
    dp_collectives = [c for c in multi.cross_collectives if "dp" in c.group_key.lower()]
    print(f"      TP collectives (AllReduce): {len(tp_collectives)}")
    print(f"      DP collectives (ZeRO-1): {len(dp_collectives)}")
    
    # Export per-rank SVG graphs
    print("\n[6] Generating per-rank SVG graphs...")
    for rank, g in graphs.items():
        output_path = f"4gpu_rank{rank}"
        visualize_graph(g, output=output_path, format="svg", show_tensors=True)
        print(f"    Saved: {output_path}.svg")
    
    # Export aggregated multi-rank graph (non-merged mode)
    print("\n[7] Generating aggregated 4-GPU SVG graph (non-merged)...")
    output_agg = "4gpu_aggregated"
    visualize_multi_rank_graph(
        multi,
        output=output_agg,
        format="svg",
        title="4-GPU Training + Inference (TP=2, DP=2)",
        show_costs=True,
        show_tensors=True,
        merged_mode=False,
    )
    print(f"    Saved: {output_agg}.svg")
    
    # Create merged version
    print("\n[8] Aggregating 4-GPU graphs (merged mode)...")
    graphs_for_merge = {}
    for rank in range(4):
        tp_group = get_tp_group(rank)
        dp_group = get_dp_group(rank)
        with build_rank(rank):
            graphs_for_merge[rank] = build_tp_dp_graph(
                rank=rank,
                tp_group=tp_group,
                dp_group=dp_group,
                hidden=256,
                inter=512,
                batch=8,
            )
    
    multi_merged = aggregate_graphs(graphs_for_merge, merge_cross_rank=True)
    print(f"    Collectives merged into single nodes spanning ranks")
    
    # Export merged aggregated graph
    print("\n[9] Generating aggregated 4-GPU SVG graph (merged)...")
    output_merged = "4gpu_merged"
    visualize_multi_rank_graph(
        multi_merged,
        output=output_merged,
        format="svg",
        title="4-GPU Training + Inference (TP=2, DP=2) - Merged",
        show_costs=True,
        show_tensors=True,
        merged_mode=True,
    )
    print(f"    Saved: {output_merged}.svg")
    
    # Export Chrome Trace JSON
    print("\n[10] Generating 4-GPU Chrome Trace JSON...")
    output_json = "4gpu_timeline.json"
    to_chrome_trace(multi, output=output_json)
    print(f"    Saved: {output_json}")
    print(f"    Open in: chrome://tracing or https://ui.perfetto.dev/")
    
    print("\n" + "=" * 60)
    print("4-GPU Example Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
    main_4gpu()
