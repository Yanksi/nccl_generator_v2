from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

from .ir import CommOp, Group, MemoryCategory, Tensor, Token, ShardSpec, tensor_replace
from .ops_schedule import SinkOp, sink  # SinkOp is a ComputeOp; re-exported here for backward compat
from .utils import bytes_of


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

@dataclass(frozen=True, eq=False, kw_only=True)
class FillOp(CommOp):
    """Generalized recv: materializes a placeholder tensor.

    In *recv mode* (src >= 0), data arrives from the network; ``group``
    must be provided to identify the communicator.
    In *local fill mode* (src == -1, source given), data is copied locally;
    ``group`` may be None.
    """
    group: Group | None = None  # required when src >= 0, optional for local fill
    src: int = -1

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.src >= 0 and self.group is None:
            raise ValueError(
                f"FillOp: 'group' is required when src >= 0 (recv mode), got src={self.src}"
            )

    @property
    def bytes(self) -> int:
        t = next(v for v in self.inputs if isinstance(v, Tensor))
        return bytes_of(t.physical_shape(), t.dtype)
    
    @property
    def kind(self) -> str:
        """Return 'recv' when receiving from network (src >= 0), 'fill' otherwise."""
        return "recv" if self.src >= 0 else "fill"

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
                grad_output, dst=self.src,
                group=self.group,
                label=_bwd_label(self.label),
                name=f"bwd_send({placeholder.name})" if placeholder.name else None,
                context=self.context,
            )
            return {placeholder: sent}


def fill(
    placeholder: Tensor,
    source: Tensor | None = None,
    *,
    group: Group | None = None,
    label: str,
    src: int = -1,
    context: str = "",
    name: str | None = None,
) -> Tensor:
    """Materialize a NOT_MATERIALIZED *placeholder* tensor.

    If *source* is given, acts as a local fill (identity / copy); ``group``
    may be omitted.
    If *source* is omitted, acts as a network recv from rank *src*; ``group``
    must be provided.

    Returns a MATERIALIZED tensor with shape/dtype from *placeholder*.
    ``requires_grad`` is inherited from *placeholder*.
    """
    inputs = (placeholder,) if source is None else (placeholder, source)
    node = FillOp(src=src, group=group, label=label, context=context, inputs=inputs)
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


@dataclass(frozen=True, eq=False, kw_only=True)
class SendOp(CommOp):
    """Send a tensor to another rank.

    Returns a NOT_MATERIALIZED tensor that aliases the input, keeping
    the original data accessible after the send.

    P2P ops identify the peer by ``dst``; ``group`` is None by default.
    """
    group: Group | None = None  # P2P: no communicator group
    dst: int = -1
    kind: str = field(default="send", init=False)

    @property
    def bytes(self) -> int:
        t = next(v for v in self.inputs if isinstance(v, Tensor))
        return bytes_of(t.physical_shape(), t.dtype)

    def vjp(self, grad_output: Tensor) -> Dict[Tensor, Tensor]:
        (x,) = [v for v in self.inputs if isinstance(v, Tensor)]
        if not x.requires_grad:
            return {}
        # Recv gradient from the rank we sent to.
        # grad_output is already a NOT_MATERIALIZED tensor, so we can recv directly into it.
        grad_x = fill(
            grad_output, src=self.dst,
            group=self.group,
            label=_bwd_label(self.label),
            name=f"bwd_recv({x.name})" if x.name else None,
            context=self.context,
        )
        return {x: grad_x}


def send(
    x: Tensor,
    *,
    dst: int,
    group: Group | None = None,
    label: str,
    context: str = "",
    name: str | None = None,
) -> Tensor:
    """Send *x* to rank *dst*.

    Returns a NOT_MATERIALIZED tensor that aliases *x*, serving as
    a dependency token while keeping the data accessible.
    """
    node = SendOp(dst=dst, group=group, label=label, context=context, inputs=(x,))
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


# -------- Collectives (TP/DP) --------

@dataclass(frozen=True, eq=False)
class AllReduceOp(CommOp):
    kind: str = field(default="allreduce", init=False)

    @property
    def bytes(self) -> int:
        t = next(v for v in self.inputs if isinstance(v, Tensor))
        return bytes_of(t.physical_shape(), t.dtype)

    def vjp(self, grad_output: Tensor) -> Dict[Tensor, Tensor]:
        # AllReduce is linear: gradient is also allreduce
        (x,) = [v for v in self.inputs if isinstance(v, Tensor)]
        if not x.requires_grad:
            return {}
        dx, _tok = allreduce(grad_output, group=self.group, label=_bwd_label(self.label), name=f"d_allreduce({x.name})", context=self.context)
        return {x: dx}

def allreduce(x: Tensor, *, group: Group, label: str, context: str = "", name: str | None = None) -> Tuple[Tensor, Token]:
    node = AllReduceOp(group=group, label=label, context=context, inputs=(x,))
    y = tensor_replace(x, producer=node, name=name, requires_grad=x.requires_grad, tp_group=group,
                        memory_category=MemoryCategory.MATERIALIZED, aliases=None)
    tok = Token(producer=node, name=(None if name is None else name + ".tok"))
    return y, tok


@dataclass(frozen=True, eq=False)
class ReduceScatterOp(CommOp):
    kind: str = field(default="reduce_scatter", init=False)

    @property
    def bytes(self) -> int:
        t = next(v for v in self.inputs if isinstance(v, Tensor))
        return bytes_of(t.physical_shape(), t.dtype)

    def vjp(self, grad_output: Tensor) -> Dict[Tensor, Tensor]:
        # ReduceScatter is linear: gradient is allgather
        (x,) = [v for v in self.inputs if isinstance(v, Tensor)]
        if not x.requires_grad:
            return {}
        dx, _tok = allgather(grad_output, group=self.group, label=_bwd_label(self.label), name=f"d_reduce_scatter({x.name})", context=self.context)
        return {x: dx}

def reduce_scatter(x: Tensor, *, group: Group, label: str, shard_axis: int = 0, context: str = "", name: str | None = None) -> Tuple[Tensor, Token]:
    """
    ReduceScatter: reduces across the group and scatters shards.

    The output tensor keeps the same logical shape as the input.
    Only the ``shard`` property changes to indicate that the tensor
    is now sharded along ``shard_axis``.
    """
    assert x.shape[shard_axis] % group.size == 0, "shape not divisible for reduce_scatter"

    node = ReduceScatterOp(group=group, label=label, context=context, inputs=(x,))
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
    shard_axis: int = 0
    kind: str = field(default="allgather", init=False)

    @property
    def bytes(self) -> int:
        t = next(v for v in self.inputs if isinstance(v, Tensor))
        # AllGather output is the full (unshard) tensor size
        return bytes_of(t.shape, t.dtype)

    def vjp(self, grad_output: Tensor) -> Dict[Tensor, Tensor]:
        # AllGather is linear: gradient is reduce_scatter
        (x,) = [v for v in self.inputs if isinstance(v, Tensor)]
        if not x.requires_grad:
            return {}
        dx, _tok = reduce_scatter(grad_output, group=self.group, shard_axis=self.shard_axis, label=_bwd_label(self.label), name=f"d_allgather({x.name})", context=self.context)
        return {x: dx}

def allgather(x_shard: Tensor, *, group: Group, label: str, context: str = "", name: str | None = None) -> Tuple[Tensor, Token]:
    """AllGather: gathers shards across the group.

    The output tensor keeps the same logical shape as the input.
    Only the ``shard`` property changes to ``replicated``.
    """
    # Derive shard_axis from the input tensor's shard spec
    shard_axis = x_shard.shard.axis if x_shard.shard.axis is not None else 0
    node = AllGatherOp(group=group, shard_axis=shard_axis, label=label, context=context, inputs=(x_shard,))
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