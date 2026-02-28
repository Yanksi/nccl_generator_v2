from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple

from .ir import CommOp, Group, MemoryCategory, Tensor, Token, ShardSpec, tensor_replace
from .ops_schedule import wait_for
from .utils import bytes_of, numel


# -------- helpers --------

def _bwd_label(label: str) -> str:
    """Derive a backward label from a forward label."""
    if ".fwd" in label:
        return label.replace(".fwd", ".bwd")
    elif ".bwd" in label:
        return label.replace(".bwd", ".fwd")
    elif label:
        return f"bwd/{label}"
    return ""


# -------- PP fill / send --------

@dataclass(frozen=True, eq=False)
class FillOp(CommOp):
    """Generalized recv: materializes a placeholder tensor.

    In *recv mode* (no source), data arrives from the network.
    In *local fill mode* (source given), data is copied locally.
    """
    bytes: int = 0
    src: int = -1
    tag: int = 0
    label: str = ""
    kind: str = field(default="fill", init=False)

    @property
    def display_kind(self) -> str:
        """Return 'recv' when no source (network recv), 'fill' otherwise."""
        has_source = sum(1 for v in self.inputs if isinstance(v, Tensor)) > 1
        return "fill" if has_source else "recv"

    def vjp(self, grad_output: Tensor) -> Dict[Tensor, Tensor]:
        tensors = [v for v in self.inputs if isinstance(v, Tensor)]
        placeholder = tensors[0]
        has_source = len(tensors) > 1

        if has_source:
            # Local fill mode: gradient passes through to source.
            source = tensors[1]
            if not source.requires_grad:
                return {}
            return {source: grad_output}
        else:
            # Recv mode: send gradient back to src rank.
            if not placeholder.requires_grad:
                return {}
            sent = send(
                grad_output, dst=self.src, bytes=self.bytes,
                tag=self.tag, label=_bwd_label(self.label),
                name=f"bwd_send({placeholder.name})" if placeholder.name else None,
            )
            return {placeholder: sent}


def fill(
    placeholder: Tensor,
    source: Tensor | None = None,
    *,
    src: int = -1,
    bytes: int = 0,
    tag: int = 0,
    label: str = "",
    name: str | None = None,
) -> Tensor:
    """Materialize a NOT_MATERIALIZED *placeholder* tensor.

    If *source* is given, acts as a local fill (identity / copy).
    If *source* is omitted, acts as a network recv from rank *src*.

    Returns a MATERIALIZED tensor with shape/dtype from *placeholder*.
    ``requires_grad`` is inherited from *placeholder*.
    """
    inputs = (placeholder,) if source is None else (placeholder, source)
    node = FillOp(bytes=bytes, src=src, tag=tag, label=label, inputs=inputs)
    return Tensor(
        shape=placeholder.shape,
        dtype=placeholder.dtype,
        producer=node,
        requires_grad=placeholder.requires_grad,
        name=name,
        shard=placeholder.shard,
        tp_group=placeholder.tp_group,
        dp_group=placeholder.dp_group,
    )


@dataclass(frozen=True, eq=False)
class SendOp(CommOp):
    """Send a tensor to another rank.

    Returns a NOT_MATERIALIZED tensor that aliases the input, keeping
    the original data accessible after the send.
    """
    bytes: int = 0
    dst: int = -1
    tag: int = 0
    label: str = ""
    kind: str = field(default="send", init=False)

    def vjp(self, grad_output: Tensor) -> Dict[Tensor, Tensor]:
        (x,) = [v for v in self.inputs if isinstance(v, Tensor)]
        if not x.requires_grad:
            return {}
        # Recv gradient from the rank we sent to.
        # Propagate shard / tp_group / dp_group so the backward placeholder
        # carries the same logical-shape + shard metadata as the forward tensor.
        bwd_placeholder = Tensor(
            shape=grad_output.shape,
            dtype=grad_output.dtype,
            memory_category=MemoryCategory.NOT_MATERIALIZED,
            requires_grad=False,
            name=f"bwd_placeholder({x.name})" if x.name else None,
            shard=grad_output.shard,
            tp_group=grad_output.tp_group,
            dp_group=grad_output.dp_group,
        )
        raw_grad = fill(
            bwd_placeholder, src=self.dst, bytes=self.bytes,
            tag=self.tag, label=_bwd_label(self.label),
            name=f"bwd_recv({x.name})" if x.name else None,
        )
        # Gate behind grad_output to maintain causal ordering:
        # the backward recv cannot complete before the forward send.
        grad_x = wait_for(
            raw_grad, grad_output,
            name=f"bwd_recv_gated({x.name})" if x.name else None,
        )
        return {x: grad_x}


def send(
    x: Tensor,
    *,
    dst: int,
    bytes: int,
    tag: int = 0,
    label: str = "",
    name: str | None = None,
) -> Tensor:
    """Send *x* to rank *dst*.

    Returns a NOT_MATERIALIZED tensor that aliases *x*, serving as
    a dependency token while keeping the data accessible.
    """
    node = SendOp(bytes=bytes, dst=dst, tag=tag, label=label, inputs=(x,))
    return Tensor(
        shape=x.shape,
        dtype=x.dtype,
        producer=node,
        requires_grad=x.requires_grad,
        name=name,
        memory_category=MemoryCategory.NOT_MATERIALIZED,
        aliases=x,
        shard=x.shard,
        tp_group=x.tp_group,
        dp_group=x.dp_group,
    )


@dataclass(frozen=True, eq=False)
class SinkOp(CommOp):
    kind: str = field(default="sink", init=False)

def sink(*deps: Token | Tensor, name: str | None = None) -> Token:
    """Create a barrier Token that depends on all *deps* (Tokens or Tensors)."""
    node = SinkOp(inputs=tuple(deps))
    tok = Token(producer=node, name=name)
    return tok


# -------- Collectives (TP/DP) --------

@dataclass(frozen=True, eq=False)
class AllReduceOp(CommOp):
    bytes: int = 0
    group: Group = Group("tp", 0, 1)
    kind: str = field(default="allreduce", init=False)

    def vjp(self, grad_output: Tensor) -> Dict[Tensor, Tensor]:
        # AllReduce is linear: gradient is also allreduce
        (x,) = [v for v in self.inputs if isinstance(v, Tensor)]
        if not x.requires_grad:
            return {}
        dx, _tok = allreduce(grad_output, group=self.group, name=f"d_allreduce({x.name})")
        return {x: dx}

def allreduce(x: Tensor, *, group: Group, bytes: int | None = None, name: str | None = None) -> Tuple[Tensor, Token]:
    b = bytes if bytes is not None else bytes_of(x.physical_shape(), x.dtype)
    node = AllReduceOp(bytes=b, group=group, inputs=(x,))
    y = tensor_replace(x, producer=node, name=name, requires_grad=x.requires_grad, tp_group=group,
                        memory_category=MemoryCategory.MATERIALIZED, aliases=None)
    tok = Token(producer=node, name=(None if name is None else name + ".tok"))
    return y, tok


@dataclass(frozen=True, eq=False)
class ReduceScatterOp(CommOp):
    bytes: int = 0
    group: Group = Group("dp", 0, 1)
    kind: str = field(default="reduce_scatter", init=False)

    def vjp(self, grad_output: Tensor) -> Dict[Tensor, Tensor]:
        # ReduceScatter is linear: gradient is allgather
        (x,) = [v for v in self.inputs if isinstance(v, Tensor)]
        if not x.requires_grad:
            return {}
        dx, _tok = allgather(grad_output, group=self.group, name=f"d_reduce_scatter({x.name})")
        return {x: dx}

def reduce_scatter(x: Tensor, *, group: Group, shard_axis: int = 0, bytes: int | None = None, name: str | None = None) -> Tuple[Tensor, Token]:
    """
    ReduceScatter: reduces across the group and scatters shards.

    The output tensor keeps the same logical shape as the input.
    Only the ``shard`` property changes to indicate that the tensor
    is now sharded along ``shard_axis``.
    """
    assert x.shape[shard_axis] % group.size == 0, "shape not divisible for reduce_scatter"

    b = bytes if bytes is not None else bytes_of(x.physical_shape(), x.dtype)
    node = ReduceScatterOp(bytes=b, group=group, inputs=(x,))
    y = tensor_replace(
        x,
        producer=node,
        name=name,
        requires_grad=x.requires_grad,
        dp_group=group,
        shard=ShardSpec("sharded", axis=shard_axis, parts=group.size),
        memory_category=MemoryCategory.MATERIALIZED,
        aliases=None,
    )
    tok = Token(producer=node, name=(None if name is None else name + ".tok"))
    return y, tok


@dataclass(frozen=True, eq=False)
class AllGatherOp(CommOp):
    bytes: int = 0
    group: Group = Group("dp", 0, 1)
    shard_axis: int = 0
    kind: str = field(default="allgather", init=False)

    def vjp(self, grad_output: Tensor) -> Dict[Tensor, Tensor]:
        # AllGather is linear: gradient is reduce_scatter
        (x,) = [v for v in self.inputs if isinstance(v, Tensor)]
        if not x.requires_grad:
            return {}
        dx, _tok = reduce_scatter(grad_output, group=self.group, shard_axis=self.shard_axis, name=f"d_allgather({x.name})")
        return {x: dx}

def allgather(x_shard: Tensor, *, group: Group, bytes: int | None = None, name: str | None = None) -> Tuple[Tensor, Token]:
    """AllGather: gathers shards across the group.

    The output tensor keeps the same logical shape as the input.
    Only the ``shard`` property changes to ``replicated``.
    """
    # Derive shard_axis from the input tensor's shard spec
    shard_axis = x_shard.shard.axis if x_shard.shard.axis is not None else 0
    b = bytes if bytes is not None else bytes_of(x_shard.shape, x_shard.dtype)
    node = AllGatherOp(bytes=b, group=group, shard_axis=shard_axis, inputs=(x_shard,))
    y = tensor_replace(
        x_shard,
        producer=node,
        name=name,
        requires_grad=x_shard.requires_grad,
        dp_group=group,
        shard=ShardSpec("replicated"),
        memory_category=MemoryCategory.MATERIALIZED,
        aliases=None,
    )
    tok = Token(producer=node, name=(None if name is None else name + ".tok"))
    return y, tok