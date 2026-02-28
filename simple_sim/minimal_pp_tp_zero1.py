from __future__ import annotations

from simple_sim import (
    Tensor, Group, ShardSpec, MemoryCategory,
    input_tensor, parameter,
    fill, send, sink,
    megatron_mlp,
    backward,
    Zero1Plan, zero1_optimizer_step,
    get_graph,
)

def build_rank_step(*, rank: int, tp_size: int, dp_size: int, pp_prev: int | None, pp_next: int | None):
    tp = Group("tp", id=0, size=tp_size)
    dp = Group("dp", id=0, size=dp_size)

    placeholder = None

    # Pipeline input
    if pp_prev is None:
        x = input_tensor((8, 4096), name="x")
    else:
        placeholder = Tensor(
            shape=(8, 4096), dtype="fp16",
            memory_category=MemoryCategory.NOT_MATERIALIZED,
            requires_grad=True, name="pp.recv.placeholder",
        )
        x = fill(placeholder, src=pp_prev, bytes=8*4096*2, tag=123, name="pp.recv.x")

    # TP MLP params (Megatron split) — full logical shapes with shard specs
    w1 = parameter((4096, 16384), name="w1", tp_group=tp, dp_group=dp, shard=ShardSpec("sharded", axis=1, parts=tp_size))
    w2 = parameter((16384, 4096), name="w2", tp_group=tp, dp_group=dp, shard=ShardSpec("sharded", axis=0, parts=tp_size))

    y = megatron_mlp(x, w1, w2, tp_group=tp, name="mlp")

    # Pipeline send (returns NOT_MATERIALIZED Tensor, aliases y)
    if pp_next is not None:
        loss = send(y, dst=pp_next, bytes=8*4096*2, tag=456, name="pp.send.y")
    else:
        loss = y

    # Backward — send.vjp auto-recvs grad; fill.vjp auto-sends grad
    wrt = [w1, w2]
    if placeholder is not None:
        wrt.append(placeholder)

    grads = backward(loss, wrt=wrt)

    # Collect backward PP send dependency (if any)
    deps: list = []
    if placeholder is not None and placeholder in grads:
        deps.append(grads[placeholder])
    step_tok = sink(*deps, name="pp.step_done") if deps else sink(name="pp.step_done")

    plan = Zero1Plan(dp_group=dp, gather_policy="eager_allgather")
    new_params, opt_tok = zero1_optimizer_step([w1, w2], grads, plan=plan, after=step_tok, name="zero1")

    g = get_graph(opt_tok)  # roots side effects
    return g

if __name__ == "__main__":
    g = build_rank_step(rank=0, tp_size=4, dp_size=8, pp_prev=None, pp_next=1)
    print(f"extracted nodes: {len(g.nodes)}")
    for n in g.nodes[:10]:
        print(n.id, getattr(n, "kind", type(n).__name__))