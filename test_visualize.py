import sys
sys.path.insert(0, '.')

from simple_sim import (
    input_tensor,
    input_factory,
    parameter,
    matmul,
    add,
    activation,
    backward,
    checkpoint,
    adam_update,
    completion,
    clear_checkpoint_registry,
    get_graph,
    print_graph_summary,
    detach,
    Tensor,
    Group,
    ShardSpec,
    MemoryCategory,
    fill,
    send,
    sink,
    wait_for,
    megatron_mlp,
    megatron_mlp_sp,
    Zero1Plan,
    zero1_optimizer_step,
    zero1_gather_params,
    param_factory,
    # PP scheduling
    PPSchedule,
    GPipeSchedule,
    OneFOneBSchedule,
)
from simple_sim.ir import tensor_replace
from simple_sim.visualize import visualize_graph


def build_training_iteration(x, w1, w2, w3, iteration, use_checkpoint=True):
    """Build one training iteration with 3-layer FFN."""
    
    if use_checkpoint:
        # Wrap first two layers in checkpoint (layers 1 and 2)
        # Layer 3 is outside the checkpoint
        with checkpoint(name=f"iter{iteration}.ckpt12") as ckpt:
            h = ckpt.enter(x)
            h = matmul(h, w1, name=f"iter{iteration}.linear1")  # Layer 1
            h = matmul(h, w2, name=f"iter{iteration}.linear2")  # Layer 2
            h = ckpt.exit(h)
    else:
        # No checkpoint - all layers are saved
        h = matmul(x, w1, name=f"iter{iteration}.linear1")
        h = matmul(h, w2, name=f"iter{iteration}.linear2")
    
    out = matmul(h, w3, name=f"iter{iteration}.linear3")  # Layer 3 (outside checkpoint)
    
    return out


def build_full_training(num_iterations=2, use_checkpoint=True):
    """Build a complete training loop with multiple iterations."""
    
    # Clear registry
    clear_checkpoint_registry()
    
    # Initial parameters (3 layers)
    w1 = parameter((8, 16), name="w1")  # Layer 1
    w2 = parameter((16, 16), name="w2")  # Layer 2
    w3 = parameter((16, 4), name="w3")   # Layer 3
    
    params = [w1, w2, w3]
    prev_x = None          # input tensor from previous iteration
    prev_done = None        # completion token from previous iteration
    
    for iteration in range(num_iterations):
        print(f"Building iteration {iteration}...")
        
        # Input for this iteration — reuse the same tensor, gated behind
        # the previous iteration's completion.
        x = input_factory(
            prev_x,
            after=prev_done,
            create_fn=lambda: input_tensor((4, 8), name="x"),
            iteration=iteration,
        )
        
        # Forward pass with 3-layer FFN
        loss = build_training_iteration(x, params[0], params[1], params[2], iteration, use_checkpoint)
        
        # Backward pass - compute gradients
        grads = backward(loss, wrt=params)
        
        # Include gradients in graph for visualization
        # graph_roots = [loss] + list(grads.values())
        
        # Apply Adam optimizer update to parameters
        new_w1 = adam_update(params[0], grads[params[0]], name=f"iter{iteration}.adam_w1")
        new_w2 = adam_update(params[1], grads[params[1]], name=f"iter{iteration}.adam_w2")
        new_w3 = adam_update(params[2], grads[params[2]], name=f"iter{iteration}.adam_w3")
        
        # Detach for next iteration (stop gradient flow)
        params = [detach(new_w1), detach(new_w2), detach(new_w3)]
        
        # Create a completion token marking the end of this iteration.
        # The next iteration's input (and params) will depend on this.
        prev_done = completion(new_w1, new_w2, new_w3,
                               name=f"iter{iteration}.done")
        prev_x = x  # remember the input tensor for reuse
        
        print(f"  Forward: x -> linear1 -> linear2 [checkpoint] -> linear3 -> loss")
        print(f"  Gradients computed and parameters updated for iteration {iteration}")
    
    # Get final graph (include all grads for visualization)
    g = get_graph(loss)
    return g


# ====================================================================
# Distributed example: TP=4, DP=4, PP=8  — single GPU, one PP stage
# ====================================================================

def build_distributed_single_gpu(
    pp_stage: int = 3,
    tp_size: int = 4,
    dp_size: int = 4,
    pp_size: int = 8,
    hidden: int = 1024,
    inter: int = 4096,
    seq: int = 2048,
    batch: int = 8,
    num_microbatches: int = 4,
    num_iterations: int = 1,
    schedule: PPSchedule | None = None,
):
    """
    Build the computation graph visible to **one GPU** in a distributed
    training setup with TP=4, DP=4, PP=8, with microbatch scheduling.

    The GPU sits at a middle pipeline stage.  Its graph contains:

    Training (per iteration, gradient accumulation over microbatches):
      For each microbatch m:
        Forward (Sequence Parallelism — activations are sequence-sharded):
          1. fill (recv) sequence-sharded activation from previous PP stage
          2. Megatron-style MLP with SP:
             AllGather → column-parallel matmul → activation → row-parallel matmul → ReduceScatter
          3. send sequence-sharded activation to next PP stage

        Backward (auto-generated by autograd via fill/send vjps):
          - send.vjp: recv gradient from next stage (fill)
          - fill.vjp: send input gradient to previous stage (send)

      Accumulate per-microbatch gradients → ZeRO-1 optimizer step

    Optimizer:
      ZeRO-1 (reduce-scatter grad → Adam shard update → allgather
      param) over the DP group.

    Inference:
      After training, run one forward pass on the updated weights (no grad).

    Args:
        schedule: Pipeline schedule to use. If None, defaults to GPipeSchedule.
                  Use OneFOneBSchedule() for 1F1B interleaved scheduling.
    """
    if schedule is None:
        schedule = GPipeSchedule()
    
    clear_checkpoint_registry()

    # ---- group handles (this GPU's local view) ----
    tp = Group("tp", id=0, size=tp_size)
    dp = Group("dp", id=0, size=dp_size)

    prev_pp_rank = pp_stage - 1          # source for fwd activation
    next_pp_rank = pp_stage + 1          # dest   for fwd activation

    # ---- parameters (TP-sharded, logical shapes) ----
    w1_col = parameter(
        (hidden, inter), name="w1_col",
        tp_group=tp, shard=ShardSpec("sharded", axis=1, parts=tp_size),
    )
    w2_row = parameter(
        (inter, hidden), name="w2_row",
        tp_group=tp, shard=ShardSpec("sharded", axis=0, parts=tp_size),
    )
    orig_params = [w1_col, w2_row]
    params = list(orig_params)

    # With SP the activation is sequence-sharded across TP ranks
    # Logical shape is the full (unsharded) activation
    tokens = seq * batch
    act_shape = (tokens, hidden)               # full logical shape
    act_shard = ShardSpec("sharded", axis=0, parts=tp_size)  # seq-sharded

    all_deps = []

    for it in range(num_iterations):
        # ==============================================================
        # Training: forward + backward per microbatch using schedule
        # ==============================================================
        
        # Storage for forward results (keyed by microbatch)
        fwd_placeholders: dict[int, Tensor] = {}
        fwd_losses: dict[int, Tensor] = {}
        
        # Accumulated gradients (in-place accumulation during backward)
        accumulated_grads: dict[Tensor, Tensor] = {}
        
        # Backward dependencies (gradient sends to prev stage)
        bwd_deps: list[Tensor] = []
        
        def do_forward(mb: int, after):
            """Execute forward pass for microbatch mb."""
            pp_label_fwd = f"iter{it}.mb{mb}.stage{pp_stage}.fwd"

            # 1. Fill (recv) sequence-sharded activation from previous PP stage
            placeholder = Tensor(
                shape=act_shape, dtype="fp16",
                memory_category=MemoryCategory.NOT_MATERIALIZED,
                requires_grad=True,
                name=f"iter{it}.mb{mb}.pp.recv_act.placeholder",
                shard=act_shard,
                tp_group=tp,
            )
            inp = placeholder
            if after is not None:
                inp = wait_for(placeholder, after, name=f"iter{it}.mb{mb}.fwd.wait_prev")
            x = fill(
                inp, src=prev_pp_rank, tag=mb,
                label=pp_label_fwd, name=f"iter{it}.mb{mb}.pp.recv_act",
            )

            # 2. Megatron MLP with Sequence Parallelism
            y = megatron_mlp_sp(x, params[0], params[1], tp_group=tp, seq_axis=0,
                                name=f"iter{it}.mb{mb}.stage.mlp")

            # 3. Send sequence-sharded activation to next PP stage
            loss_mb = send(
                y, dst=next_pp_rank, tag=mb,
                label=pp_label_fwd, name=f"iter{it}.mb{mb}.pp.send_act",
            )
            
            fwd_placeholders[mb] = placeholder
            fwd_losses[mb] = loss_mb
            return completion(loss_mb, name=f"iter{it}.mb{mb}.fwd.done")
        
        def do_backward(mb: int, after):
            """Execute backward pass for microbatch mb, accumulating gradients in-place."""
            placeholder = fwd_placeholders[mb]
            loss_mb = fwd_losses[mb]
            
            # Ensure backward computation waits on schedule ordering
            # Gate the loss so all backward ops transitively depend on `after`
            if after is not None:
                loss_mb = wait_for(loss_mb, after, name=f"iter{it}.mb{mb}.bwd.wait_sched")
            
            grads_mb = backward(loss_mb, wrt=params + [placeholder])
            
            # Accumulate parameter gradients in-place
            for p in params:
                grad = grads_mb[p]
                if p in accumulated_grads:
                    # Add to existing accumulated gradient
                    accumulated_grads[p] = add(accumulated_grads[p], grad, 
                                               name=f"iter{it}.acc_grad.{p.name}.mb{mb}")
                else:
                    # First microbatch: store gradient directly
                    accumulated_grads[p] = grad
            
            # Collect gradient send to prev stage (for PP backward)
            if placeholder in grads_mb:
                bwd_deps.append(grads_mb[placeholder])
            
            return completion(*grads_mb.values(), name=f"iter{it}.mb{mb}.bwd.done")
        
        # Execute schedule - scheduler handles ordering and creates dependency edges
        schedule.execute(num_microbatches, pp_stage, pp_size, do_forward, do_backward)

        # ---- ZeRO-1 optimizer step (DP collectives) ----
        new_params, opt_done_tok = zero1_optimizer_step(
            params, accumulated_grads, plan=Zero1Plan(dp_group=dp), name=f"iter{it}.zero1",
        )

        all_deps.extend(bwd_deps)
        all_deps.append(opt_done_tok)

        # Detach params for next iteration
        params = [detach(p, name=p.name) for p in new_params]

    # ==============================================================
    # Inference: one forward pass on updated weights (no grad)
    # ==============================================================
    inf_placeholder = Tensor(
        shape=act_shape, dtype="fp16",
        memory_category=MemoryCategory.NOT_MATERIALIZED,
        requires_grad=False,
        name="inf.pp.recv_act.placeholder",
        shard=act_shard,
        tp_group=tp,
    )
    inf_x = fill(
        inf_placeholder, src=prev_pp_rank, tag=100,
        label="inf.fwd", name="inf.pp.recv_act",
    )
    inf_y = megatron_mlp_sp(inf_x, params[0], params[1], tp_group=tp, seq_axis=0,
                            name="inf.stage.mlp")
    inf_sent = send(
        inf_y, dst=next_pp_rank, tag=100,
        label="inf.fwd", name="inf.pp.send_act",
    )

    # ---- collect all graph roots ----
    done = sink(*all_deps, inf_sent, name="stage.done")
    g = get_graph(done)
    return g


# ====================================================================
# DP-only example with gather_on_demand ZeRO-1
# ====================================================================

def build_dp_gather_on_demand(
    dp_size: int = 4,
    hidden: int = 256,
    inter: int = 1024,
    batch: int = 32,
    num_iterations: int = 2,
):
    """
    DP-only training with ZeRO-1 in gather_on_demand mode.

    Each iteration:
      1. allgather params (deferred from previous optimizer step)
      2. forward:  matmul → activation → matmul
      3. backward
      4. zero1_optimizer_step (reduce-scatter grad → Adam shard)
         — returns DP-sharded params, NO allgather yet

    The allgather is deferred to step 1 of the *next* iteration,
    giving the scheduler room to overlap it with other work.
    """
    clear_checkpoint_registry()

    dp = Group("dp", id=0, size=dp_size)
    plan = Zero1Plan(dp_group=dp, gather_policy="gather_on_demand")

    # ---- initial parameters ----
    w1 = parameter((hidden, inter), name="w1", dp_group=dp)
    w2 = parameter((inter, hidden), name="w2", dp_group=dp)
    orig_params = [w1, w2]          # keep for metadata restoration
    params = list(orig_params)

    prev_tok = None
    all_tokens = []

    for it in range(num_iterations):
        # If we have sharded params from a previous optimizer step,
        # allgather them now ("gather on demand").
        if prev_tok is not None:
            params, gather_tok = zero1_gather_params(
                params, orig_params,
                plan=plan,
                name=f"iter{it}.zero1",
            )
            all_tokens.append(gather_tok)

        # Detach params (stop gradient flow between iterations)
        params = [
            detach(p, name=f"iter{it}.{p.name}" if p.name else None)
            for p in params
        ]

        # ---- forward ----
        x = input_tensor((batch, hidden), name=f"iter{it}.x")
        h = matmul(x, params[0], name=f"iter{it}.linear1")
        h = activation(h, name=f"iter{it}.act")
        loss = matmul(h, params[1], name=f"iter{it}.linear2")

        # ---- backward ----
        grads = backward(loss, wrt=params)

        # ---- ZeRO-1 optimizer (gather_on_demand) ----
        params, opt_tok = zero1_optimizer_step(
            params, grads,
            plan=plan,
            name=f"iter{it}.zero1",
        )
        all_tokens.append(opt_tok)
        prev_tok = opt_tok

    done = sink(*all_tokens, name="training.done")
    g = get_graph(done)
    return g


if __name__ == "__main__":
    # # Build training with checkpoints
    # print("=" * 60)
    # print("WITH GRADIENT CHECKPOINTING")
    # print("Checkpoint covers: linear1 + linear2")
    # print("Outside checkpoint: linear3")
    # print("=" * 60)
    # g_ckpt = build_full_training(num_iterations=2, use_checkpoint=True)
    
    # print(f"\nGraph has {len(g_ckpt.nodes)} nodes")
    # print_graph_summary(g_ckpt)
    
    # visualize_graph(g_ckpt, output='graph_3layer_checkpoint', format='svg', show_tensors=True)
    # print("\nSaved to graph_3layer_checkpoint.svg")
    
    # # Build training without checkpoints
    # print("\n" + "=" * 60)
    # print("WITHOUT GRADIENT CHECKPOINTING")
    # print("All layers saved (no checkpoint)")
    # print("=" * 60)
    # g_no_ckpt = build_full_training(num_iterations=2, use_checkpoint=False)
    
    # print(f"\nGraph has {len(g_no_ckpt.nodes)} nodes")
    # print_graph_summary(g_no_ckpt)
    
    # visualize_graph(g_no_ckpt, output='graph_3layer_no_checkpoint', format='svg', show_tensors=True)
    # print("\nSaved to graph_3layer_no_checkpoint.svg")

    # ================================================================
    # Distributed example: TP=4, DP=4, PP=8  (single GPU, middle stage)
    # with microbatch scheduling, gradient accumulation, and inference
    # ================================================================
    print("\n" + "=" * 60)
    print("DISTRIBUTED (GPipe): TP=4, DP=4, PP=3  (single GPU, PP stage 1)")
    print("  Megatron MLP with TP + Sequence Parallelism")
    print("  ZeRO-1 optimizer (DP reduce-scatter + allgather)")
    print("  PP send/recv for activations & gradients")
    print("  4 microbatches with gradient accumulation")
    print("  Schedule: GPipe (all forwards, then all backwards)")
    print("=" * 60)
    gpipe_schedule = GPipeSchedule()
    print(f"Schedule steps: {gpipe_schedule.generate(4, pp_stage=1, pp_size=3)}")
    g_gpipe = build_distributed_single_gpu(num_microbatches=4, num_iterations=1, schedule=gpipe_schedule, pp_stage=1, pp_size=3)
    print(f"\nGraph has {len(g_gpipe.nodes)} nodes")
    print_graph_summary(g_gpipe)
    visualize_graph(g_gpipe, output='graph_distributed_gpipe', format='svg', show_tensors=True)
    print("\nSaved to graph_distributed_gpipe.svg")

    print("\n" + "=" * 60)
    print("DISTRIBUTED (GPipe): TP=4, DP=4, PP=3  (single GPU, PP stage 1)")
    print("  Same config as above, but with 1F1B interleaved schedule")
    print("  Schedule: 1F1B (warmup → steady 1F1B → drain)")
    print("=" * 60)
    one_f_one_b_schedule = OneFOneBSchedule()
    print(f"Schedule steps: {one_f_one_b_schedule.generate(4, pp_stage=1, pp_size=3)}")
    g_1f1b = build_distributed_single_gpu(num_microbatches=4, num_iterations=1, schedule=one_f_one_b_schedule, pp_stage=1, pp_size=3)
    print(f"\nGraph has {len(g_1f1b.nodes)} nodes")
    print_graph_summary(g_1f1b)
    visualize_graph(g_1f1b, output='graph_distributed_1f1b', format='svg', show_tensors=True)
    print("\nSaved to graph_distributed_1f1b.svg")

    # # ================================================================
    # # DP-only with gather_on_demand ZeRO-1
    # # ================================================================
    # print("\n" + "=" * 60)
    # print("DP-ONLY: gather_on_demand ZeRO-1, 2 iterations")
    # print("  Optimizer returns DP-sharded params")
    # print("  Allgather deferred to start of next iteration")
    # print("=" * 60)
    # g_dp = build_dp_gather_on_demand(num_iterations=2)
    # print(f"\nGraph has {len(g_dp.nodes)} nodes")
    # print_graph_summary(g_dp)
    # visualize_graph(g_dp, output='graph_dp_gather_on_demand', format='svg', show_tensors=True)
    # print("\nSaved to graph_dp_gather_on_demand.svg")
