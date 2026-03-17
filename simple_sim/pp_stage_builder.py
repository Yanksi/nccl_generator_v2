"""Generalized single-GPU PP stage graph builder.

This module provides :func:`build_pp_stage_graph`, a flexible orchestration
shell that captures all the boilerplate for pipeline-parallel training on one
GPU (recv activation → user compute → send activation → backward →
ZeRO-1 → inference) while accepting the actual forward computation and
parameter/activation metadata as caller-supplied inputs.

Model spec contract
-------------------
``model_spec`` is a ``dict[str, Callable[[Group, Group], Any]]`` with two
kinds of entries:

``"_input"`` (**required**)
    Factory called as ``f(tp_group, dp_group)`` that returns a 3-tuple
    ``(shape, dtype, shard)`` describing the sequence-sharded PP activation:

    * ``shape`` — full logical shape, e.g. ``(tokens, hidden)``
    * ``dtype`` — string dtype, e.g. ``"fp16"``
    * ``shard`` — :class:`~simple_sim.ShardSpec` for the sequence dimension

every other key
    Parameter factory called as ``f(tp_group, dp_group) -> Tensor``
    (typically wrapping :func:`~simple_sim.parameter`).  Insertion order
    determines the parameter list used by autograd and ZeRO-1.

Forward function signature
--------------------------
``forward_fn`` is called as::

    y = forward_fn(x, params_dict, ctx, *, name)

where

* ``x`` — input activation ``Tensor`` (received from previous PP stage,
  or produced by ``first_stage_input_fn`` on the first stage)
* ``params_dict`` — ``dict[str, Tensor]`` mapping parameter names to their
  current-iteration ``Tensor`` objects
* ``ctx`` — ``dict[str, ParallelDim]`` with keys ``"tp"``, ``"dp"``,
  ``"pp"``, each a :class:`ParallelDim` holding ``.rank``, ``.size``,
  and ``.group``.  The forward function can derive boundary flags such as
  ``is_last_stage = ctx["pp"].rank == ctx["pp"].size - 1`` itself.
* ``name`` — string prefix passed to sub-operations for readable node names

The function must return a single ``Tensor`` ``y`` representing the output
activation to be sent to the next PP stage (or the loss on the last stage).

``first_stage_input_fn``, when provided, is called as::

    x = first_stage_input_fn(params_dict, ctx, *, name)

with the same ``params_dict`` and ``ctx``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .ir import Group, MemoryCategory, ShardSpec, Tensor, Token
from .ops_comm import fill, send
from .ops_compute import add
from .ops_schedule import completion, detach, sink, wait_for
from .autograd import backward
from .extract import get_graph
from .checkpoint import clear_checkpoint_registry
from .zero1 import Zero1Plan, zero1_optimizer_step
from .pp_schedule import GPipeSchedule, PPSchedule


@dataclass(frozen=True)
class ParallelDim:
    """Parallelism coordinate for one dimension (TP, DP, or PP).

    Bundles together the three pieces of information a forward function
    needs about a single parallelism axis:

    * ``rank``  — this device's index within the dimension
    * ``size``  — total number of devices in the dimension
    * ``group`` — the :class:`~simple_sim.Group` communicator handle
    """
    rank: int
    size: int
    group: Group


# ---------------------------------------------------------------------------
# Type aliases (documentation only; not enforced at runtime)
# ---------------------------------------------------------------------------

# ctx passed to forward_fn / first_stage_input_fn
# Keys: "tp", "dp", "pp"  (and any extra dims the caller adds)
_ParallelCtx = dict[str, ParallelDim]

# (tp_group, dp_group) -> Tensor
_ParamFactory = Callable[[Group, Group], Tensor]
# (tp_group, dp_group) -> (shape, dtype, ShardSpec)
_InputFactory = Callable[[Group, Group], tuple[tuple[int, ...], str, ShardSpec]]
# forward_fn(x, params_dict, ctx, *, name) -> Tensor
_ForwardFn = Callable[..., Tensor]


def build_pp_stage_graph(
    forward_fn: _ForwardFn,
    model_spec: dict[str, _ParamFactory | _InputFactory],
    *,
    device_id: int = 0,
    tp_size: int = 4,
    dp_size: int = 4,
    pp_size: int = 8,
    num_microbatches: int = 4,
    num_iterations: int = 1,
    schedule: PPSchedule | None = None,
    first_stage_input_fn: Callable[..., Tensor] | None = None,
    include_inference: bool = True,
) -> Any:
    """Build the computation graph visible to one GPU in a distributed
    pipeline-parallel training setup.

    All PP/TP/DP scheduling boilerplate is handled here.  The caller supplies
    only the forward computation and parameter/activation shapes via
    ``forward_fn`` and ``model_spec`` respectively.

    The returned graph covers:

    * **Training** (``num_iterations`` iterations, each with
      ``num_microbatches`` microbatches):

      * Per microbatch: PP recv (skipped on first stage) → ``forward_fn`` →
        PP send (skipped on last stage)
      * Backward via autograd (PP send/recv VJPs are included automatically)
      * Gradient accumulation across microbatches
      * ZeRO-1 optimizer step over the DP group

    * **Inference** (only when ``include_inference=True``): one forward pass
      on the updated weights (no grad).

    Parameters
    ----------
    forward_fn:
        Callable with signature
        ``y = forward_fn(x, params_dict, ctx, *, name)``.
        See module docstring for details.  On the **last** PP stage the
        returned tensor is treated as the loss (no send follows).
    model_spec:
        Dict mapping ``"_input"`` (required) and parameter names to factory
        callables.  See module docstring for the full contract.
    device_id:
        Global device index in ``[0, tp_size * dp_size * pp_size)``.
    tp_size, dp_size, pp_size:
        Tensor / data / pipeline parallelism degrees.
    num_microbatches:
        Microbatches per training iteration.
    num_iterations:
        Number of training iterations to unroll.
    schedule:
        Pipeline schedule.  Defaults to :class:`GPipeSchedule` if ``None``.
    first_stage_input_fn:
        Optional callable with signature
        ``x = first_stage_input_fn(params_dict, ctx, *, name)`` used
        **only on the first PP stage** (``pp_stage == 0``) to produce the
        input activation instead of a PP recv.  Receives the same ``ctx``
        dict as ``forward_fn``.  If ``None``, falls back to
        :func:`~simple_sim.input_tensor` with the shape from
        ``model_spec["_input"]``.
    include_inference:
        If ``True`` (default), append a no-grad inference forward pass after
        training.  Set to ``False`` to omit it (e.g. when building a
        training-only graph).

    Returns
    -------
    ExtractedGraph
        The complete computation graph rooted at the ``stage.done`` sink.
    """
    if schedule is None:
        schedule = GPipeSchedule()

    total_devices = tp_size * dp_size * pp_size
    assert 0 <= device_id < total_devices, (
        f"device_id must be in [0, {total_devices - 1}]"
    )
    pp_stage = device_id // (tp_size * dp_size)
    local_id = device_id % (tp_size * dp_size)
    tp_rank = local_id % tp_size   # TP rank within this GPU's TP group
    dp_rank = local_id // tp_size  # DP rank within this GPU's DP group

    # PP boundary flags (used internally to control recv/send)
    is_first_stage = pp_stage == 0
    is_last_stage = pp_stage == pp_size - 1

    clear_checkpoint_registry()

    # ---- group handles (this GPU's local view) ----
    # TP group: tp_size GPUs sharing the same pp_stage and dp_rank
    tp = Group("tp", size=tp_size, name=f"pp{pp_stage}_dp{dp_rank}", self_rank=tp_rank)
    # DP group: dp_size GPUs sharing the same pp_stage and tp_rank
    dp = Group("dp", size=dp_size, name=f"pp{pp_stage}_tp{tp_rank}", self_rank=dp_rank)
    # PP group: pp_size GPUs sharing the same tp_rank and dp_rank
    pp = Group("pp", size=pp_size, name=f"tp{tp_rank}_dp{dp_rank}", self_rank=pp_stage)

    prev_pp_rank = pp_stage - 1   # source for fwd activation (invalid when first stage)
    next_pp_rank = pp_stage + 1   # dest   for fwd activation (invalid when last stage)

    # ---- parallelism context dict passed to forward_fn / first_stage_input_fn ----
    ctx: _ParallelCtx = {
        "tp": ParallelDim(rank=tp_rank, size=tp_size, group=tp),
        "dp": ParallelDim(rank=dp_rank, size=dp_size, group=dp),
        "pp": ParallelDim(rank=pp_stage, size=pp_size, group=pp),
    }

    # ---- activation spec from model_spec["_input"] ----
    assert "_input" in model_spec, (
        "model_spec must contain an '_input' key whose factory returns "
        "(shape, dtype, ShardSpec)"
    )
    act_shape, act_dtype, act_shard = model_spec["_input"](tp, dp)

    # ---- parameters (constructed in insertion order) ----
    param_keys: list[str] = [k for k in model_spec if k != "_input"]
    params_list: list[Tensor] = [model_spec[k](tp, dp) for k in param_keys]
    params_dict: dict[str, Tensor] = dict(zip(param_keys, params_list))

    all_deps: list[Tensor] = []
    iter_start_tok: Token | None = Token(producer=None, name="training_start")

    for it in range(num_iterations):
        # ==============================================================
        # Training: forward + backward per microbatch using schedule
        # ==============================================================

        # Storage for forward results keyed by microbatch index
        # fwd_placeholders[mb] is None when is_first_stage (no PP recv)
        fwd_placeholders: dict[int, Tensor | None] = {}
        fwd_losses: dict[int, Tensor] = {}

        # Accumulated gradients (in-place accumulation during backward)
        accumulated_grads: dict[Tensor, Tensor] = {}

        # Backward dependencies (gradient sends to prev stage)
        bwd_deps: list[Tensor] = []
        # Forward send tokens — collected for end-of-iteration sink only
        fwd_sends: list[Tensor] = []

        def do_forward(mb: int, after):
            """Execute forward pass for microbatch *mb*."""
            pp_label_fwd = f"iter{it}.mb{mb}.stage{pp_stage}.fwd"

            # 1. Obtain input activation.
            if is_first_stage:
                # First stage: generate input locally (no PP recv).
                if first_stage_input_fn is not None:
                    x = first_stage_input_fn(
                        params_dict, ctx,
                        name=f"iter{it}.mb{mb}.input",
                    )
                else:
                    from .ops_compute import input_tensor as _input_tensor
                    x = _input_tensor(
                        act_shape, dtype=act_dtype,
                        name=f"iter{it}.mb{mb}.input",
                    )
                placeholder = None
                # Apply per-microbatch schedule ordering.
                x_sched = x
                if after is not None:
                    x_sched = wait_for(x, after, name=f"iter{it}.mb{mb}.fwd.wait_prev")
            else:
                # Middle / last stage: recv activation from previous PP stage.
                # Gate the recv on the iteration-start token only (coarse
                # ordering) so the network recv can proceed independently of the
                # per-microbatch scheduling chain.
                placeholder = Tensor(
                    shape=act_shape, dtype=act_dtype,
                    memory_category=MemoryCategory.NOT_MATERIALIZED,
                    requires_grad=True,
                    name=f"iter{it}.mb{mb}.pp.recv_act.placeholder",
                    shard=act_shard,
                    tp_group=tp,
                )
                inp = placeholder
                if iter_start_tok is not None:
                    inp = wait_for(
                        placeholder, iter_start_tok,
                        name=f"iter{it}.mb{mb}.fwd.iter_start",
                    )
                x = fill(
                    inp, src=prev_pp_rank, group=pp,
                    label=pp_label_fwd, name=f"iter{it}.mb{mb}.pp.recv_act",
                    context="pp"
                )
                # Apply per-microbatch schedule ordering to compute, not recv.
                x_sched = x
                if after is not None:
                    x_sched = wait_for(x, after, name=f"iter{it}.mb{mb}.fwd.wait_prev")

            # 2. User-supplied forward computation.
            y = forward_fn(
                x_sched, params_dict, ctx,
                name=f"iter{it}.mb{mb}.stage.fwd",
            )

            # 3. Send output to next PP stage, or keep as loss on last stage.
            if is_last_stage:
                # Last stage: y is the loss — no PP send.
                loss_mb = y
            else:
                # Middle / first (non-last) stage: send to next stage.
                # Collected for end-of-iteration sink; NOT in the scheduling
                # token so downstream microbatch compute doesn't wait on it.
                loss_mb = send(
                    y, dst=next_pp_rank, group=pp,
                    label=pp_label_fwd, name=f"iter{it}.mb{mb}.pp.send_act",
                    context="pp"
                )
                fwd_sends.append(loss_mb)

            fwd_placeholders[mb] = placeholder  # None for first stage
            fwd_losses[mb] = loss_mb
            # Return completion over compute output only (not the send).
            return completion(y, name=f"iter{it}.mb{mb}.fwd.done")

        def do_backward(mb: int, after):
            """Execute backward pass for microbatch *mb*, accumulating grads."""
            placeholder = fwd_placeholders[mb]  # None on first stage
            loss_mb = fwd_losses[mb]

            # Run backward ungated — grad-recv (SendOp.vjp) and grad-send
            # (FillOp.vjp) are purely network-ordered; schedule ordering is
            # applied below to param grads only, keeping send/recv out of the
            # inter-microbatch dependency chain.
            wrt = params_list + ([placeholder] if placeholder is not None else [])
            grads_mb = backward(loss_mb, wrt=wrt)

            # Collect gradient send to prev stage for end-of-iteration sink.
            # (Not applicable on first stage — no previous stage to send to.)
            if placeholder is not None and placeholder in grads_mb:
                bwd_deps.append(grads_mb[placeholder])

            # Apply schedule ordering gate to param grads only.
            if after is not None:
                gated_param_grads = {
                    p: wait_for(
                        grads_mb[p], after,
                        name=f"iter{it}.mb{mb}.bwd.gate.{p.name}",
                    )
                    for p in params_list if p in grads_mb
                }
            else:
                gated_param_grads = {
                    p: grads_mb[p] for p in params_list if p in grads_mb
                }

            # Accumulate parameter gradients in-place.
            for p in params_list:
                grad = gated_param_grads[p]
                if p in accumulated_grads:
                    accumulated_grads[p] = add(
                        accumulated_grads[p], grad,
                        name=f"iter{it}.acc_grad.{p.name}.mb{mb}",
                    )
                else:
                    accumulated_grads[p] = grad

            return completion(
                *gated_param_grads.values(),
                name=f"iter{it}.mb{mb}.bwd.done",
            )

        # Execute schedule — creates all dependency edges between microbatches.
        schedule.execute(num_microbatches, pp_stage, pp_size, do_forward, do_backward)

        # ---- ZeRO-1 optimizer step (DP collectives) ----
        new_params_list, opt_done_tok = zero1_optimizer_step(
            params_list, accumulated_grads,
            plan=Zero1Plan(dp_group=dp),
            name=f"iter{it}.zero1",
        )

        all_deps.extend(fwd_sends)
        all_deps.extend(bwd_deps)
        all_deps.append(opt_done_tok)

        # Gate next iteration's PP recvs on this iteration finishing.
        iter_start_tok = opt_done_tok

        # Detach params for next iteration (stop gradient flow).
        params_list = [detach(p, name=p.name) for p in new_params_list]
        params_dict = dict(zip(param_keys, params_list))

    # ==============================================================
    # Inference: one forward pass on updated weights (no grad)
    # ==============================================================
    if include_inference:
        if is_first_stage:
            if first_stage_input_fn is not None:
                inf_x = first_stage_input_fn(
                    params_dict, ctx, name="inf.input",
                )
            else:
                from .ops_compute import input_tensor as _input_tensor
                inf_x = _input_tensor(
                    act_shape, dtype=act_dtype, name="inf.input",
                )
        else:
            inf_placeholder = Tensor(
                shape=act_shape, dtype=act_dtype,
                memory_category=MemoryCategory.NOT_MATERIALIZED,
                requires_grad=False,
                name="inf.pp.recv_act.placeholder",
                shard=act_shard,
                tp_group=tp,
            )
            inf_x = fill(
                inf_placeholder, src=prev_pp_rank, group=pp,
                label="inf.fwd", name="inf.pp.recv_act",
                context="pp"
            )

        inf_y = forward_fn(inf_x, params_dict, ctx, name="inf.stage.fwd")

        if is_last_stage:
            all_deps.append(inf_y)
        else:
            inf_sent = send(
                inf_y, dst=next_pp_rank, group=pp,
                label="inf.fwd", name="inf.pp.send_act",
                context="pp"
            )
            all_deps.append(inf_sent)

    # ---- collect all graph roots ----
    done = sink(*all_deps, name="stage.done")
    g = get_graph(done)
    return g
