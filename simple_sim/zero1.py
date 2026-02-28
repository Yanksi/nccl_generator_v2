from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .ir import Group, MemoryCategory, Tensor, Token, ShardSpec, tensor_replace
from .ops_comm import allgather, reduce_scatter, sink
from .ops_compute import adam_update_shard


@dataclass(frozen=True)
class Zero1Plan:
    dp_group: Group
    gather_policy: str = "eager_allgather"  # or "gather_on_demand"

def zero1_optimizer_step(
    params: Sequence[Tensor],
    grads: Dict[Tensor, Tensor],
    *,
    plan: Zero1Plan,
    after: Optional[Token] = None,
    name: str = "zero1",
) -> Tuple[List[Tensor], Token]:
    """
    ZeRO-1 optimizer step.

    For every parameter:
      1. reduce_scatter the gradient over the DP group → grad shard
      2. Adam update on the shard

    Then, depending on ``plan.gather_policy``:

    * **eager_allgather** – immediately allgather the updated shard back
      to a full (DP-replicated) parameter.  The returned tensors are
      ready to use in the next forward pass.

    * **gather_on_demand** – return the DP-sharded updated parameter.
      The caller is responsible for calling :func:`zero1_gather_params`
      before the next forward pass to allgather them, giving the caller
      control over *when* the communication happens.

    Returns ``(new_params, done_token)``.
    """
    effect_tokens: List[Token] = []
    new_params: List[Tensor] = []

    # order dependency: include `after` by sinking it into the output token
    if after is not None:
        effect_tokens.append(after)

    for p in params:
        g = grads[p]
        g_shard, tok_rs = reduce_scatter(
            g,
            group=plan.dp_group,
            shard_axis=0,
            name=f"{name}.rs_grad.{p.name}",
        )
        effect_tokens.append(tok_rs)

        p_shard_new = adam_update_shard(p, g_shard, name=f"{name}.adam_shard.{p.name}")

        if plan.gather_policy == "eager_allgather":
            p_new, tok_ag = allgather(
                p_shard_new,
                group=plan.dp_group,
                name=f"{name}.ag_param.{p.name}",
            )
            effect_tokens.append(tok_ag)
            # Restore the original param's metadata (TP shard, groups, etc.)
            # The allgather undoes the DP reduce-scatter, but the param may
            # still be TP-sharded, so we restore p.shard rather than forcing
            # "replicated".
            p_new = tensor_replace(
                p_new,
                requires_grad=True,
                dp_group=p.dp_group,
                tp_group=p.tp_group,
                shard=p.shard,
                name=p.name,
            )
            new_params.append(p_new)
        else:
            # gather_on_demand: return DP-sharded param.
            # Keep the DP-shard spec from the reduce_scatter (axis=0)
            # but preserve tp_group/dp_group so zero1_gather_params can
            # restore the original TP shard later.
            p_shard_new = tensor_replace(
                p_shard_new,
                requires_grad=True,
                dp_group=p.dp_group,
                tp_group=p.tp_group,
                name=p.name,
            )
            new_params.append(p_shard_new)

    done = sink(*effect_tokens, name=f"{name}.done")
    return new_params, done


def zero1_gather_params(
    sharded_params: Sequence[Tensor],
    original_params: Sequence[Tensor],
    *,
    plan: Zero1Plan,
    name: str = "zero1",
) -> Tuple[List[Tensor], Token]:
    """
    Allgather DP-sharded params produced by ``gather_on_demand`` mode.

    Call this before using the params in the next forward pass.  Each
    param is allgathered over the DP group, and the original param's
    shard / group metadata is restored.

    Args:
        sharded_params: DP-sharded params from ``zero1_optimizer_step``
            with ``gather_policy="gather_on_demand"``.
        original_params: The original parameters (from the same
            ``zero1_optimizer_step`` call) used to restore TP shard
            metadata after the allgather.
        plan: The same ``Zero1Plan`` used for the optimizer step.
        name: Name prefix for the graph nodes.

    Returns ``(gathered_params, done_token)``.
    """
    assert len(sharded_params) == len(original_params), (
        f"sharded_params ({len(sharded_params)}) and "
        f"original_params ({len(original_params)}) must have the same length"
    )

    effect_tokens: List[Token] = []
    gathered: List[Tensor] = []

    for p_shard, p_orig in zip(sharded_params, original_params):
        p_full, tok_ag = allgather(
            p_shard,
            group=plan.dp_group,
            name=f"{name}.ag_param.{p_shard.name}",
        )
        effect_tokens.append(tok_ag)

        # Restore original param metadata (TP shard spec, groups, etc.)
        p_full = tensor_replace(
            p_full,
            requires_grad=True,
            dp_group=p_orig.dp_group,
            tp_group=p_orig.tp_group,
            shard=p_orig.shard,
            name=p_orig.name,
        )
        gathered.append(p_full)

    done = sink(*effect_tokens, name=f"{name}.gather_done")
    return gathered, done