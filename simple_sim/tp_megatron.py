from __future__ import annotations

from typing import Tuple

from .ir import Group, ShardSpec, Tensor, tensor_replace
from .ops_compute import activation, matmul
from .ops_comm import allreduce, allgather, reduce_scatter


def tp_column_linear(x: Tensor, w_col: Tensor, *, tp_group: Group, name: str) -> Tensor:
    """
    Column-parallel linear: W is sharded on output features (axis=1).
    Output is sharded.
    """
    assert w_col.tp_group == tp_group, "w_col.tp_group must be set"
    y = matmul(x, w_col, name=name)
    return tensor_replace(y, tp_group=tp_group, shard=ShardSpec("sharded", axis=1, parts=tp_group.size))

def tp_row_linear(x_part: Tensor, w_row: Tensor, *, tp_group: Group, name: str) -> Tensor:
    """
    Row-parallel linear: W sharded on input features (axis=0).
    Local matmul yields partial sum in full output space; then allreduce to sum across TP ranks.
    """
    assert x_part.tp_group == tp_group, "x_part.tp_group must be set"
    assert w_row.tp_group == tp_group, "w_row.tp_group must be set"

    z_partial = matmul(x_part, w_row, name=name + ".partial")
    z, _tok = allreduce(z_partial, group=tp_group, name=name + ".allreduce")
    return tensor_replace(z, tp_group=tp_group, shard=ShardSpec("replicated"))

def megatron_mlp(x: Tensor, w1_col: Tensor, w2_row: Tensor, *, tp_group: Group, name: str = "mlp") -> Tensor:
    """
    Megatron-style MLP with Tensor Parallelism (no sequence parallelism).
    
    Communication: AllReduce after row-parallel linear.
    Input/output activations are replicated across TP ranks.
    """
    h = tp_column_linear(x, w1_col, tp_group=tp_group, name=name + ".fc1")
    h = activation(h, name=name + ".act")
    y = tp_row_linear(h, w2_row, tp_group=tp_group, name=name + ".fc2")
    return y


# ============================================================================
# Sequence Parallelism (SP) + Tensor Parallelism (TP)
# ============================================================================
# With SP, activations are sharded along the sequence dimension (axis=0 or 1).
# This replaces AllReduce with AllGather + ReduceScatter, which is more memory
# efficient as activations remain sharded between the two linears.
#
# Pattern:
#   x (seq-sharded) -> AllGather -> matmul(w1_col) -> act -> matmul(w2_row) -> ReduceScatter -> y (seq-sharded)
# ============================================================================

def sp_column_linear(x_seq_shard: Tensor, w_col: Tensor, *, tp_group: Group, seq_axis: int = 0, name: str) -> Tensor:
    """
    Column-parallel linear with Sequence Parallelism.
    
    Input is sharded along sequence dimension. We AllGather to get the full sequence,
    then do the column-parallel matmul.
    
    Args:
        x_seq_shard: Input tensor sharded along seq_axis (logical shape is full)
        w_col: Column-parallel weight (logical shape is full)
        tp_group: Tensor parallel group
        seq_axis: Axis along which sequence is sharded (default: 0)
        name: Operation name
    
    Returns:
        Output tensor with full sequence, sharded on hidden dimension
    """
    assert w_col.tp_group == tp_group, "w_col.tp_group must be set"
    
    # AllGather to get full sequence on all ranks (just changes shard spec)
    x_full, _tok = allgather(
        x_seq_shard, 
        group=tp_group, 
        name=name + ".allgather"
    )
    x_full = tensor_replace(x_full, tp_group=tp_group, shard=ShardSpec("replicated"))
    
    # Column-parallel matmul
    y = matmul(x_full, w_col, name=name + ".matmul")
    return tensor_replace(y, tp_group=tp_group, shard=ShardSpec("sharded", axis=-1, parts=tp_group.size))


def sp_row_linear(x_part: Tensor, w_row: Tensor, *, tp_group: Group, seq_axis: int = 0, name: str) -> Tensor:
    """
    Row-parallel linear with Sequence Parallelism.
    
    Does row-parallel matmul, then ReduceScatter to both sum partial results
    and shard the output along the sequence dimension.
    
    Args:
        x_part: Input tensor sharded on hidden dimension [seq, hidden/TP]
        w_row: Row-parallel weight [hidden/TP, out_features]
        tp_group: Tensor parallel group
        seq_axis: Axis along which to shard output sequence (default: 0)
        name: Operation name
    
    Returns:
        Output tensor sharded along sequence dimension [seq/TP, out_features]
    """
    assert x_part.tp_group == tp_group, "x_part.tp_group must be set"
    assert w_row.tp_group == tp_group, "w_row.tp_group must be set"
    
    # Row-parallel matmul produces partial sum
    z_partial = matmul(x_part, w_row, name=name + ".partial")
    
    # ReduceScatter: sum across TP ranks AND shard along sequence dimension
    z_seq_shard, _tok = reduce_scatter(
        z_partial, 
        group=tp_group, 
        shard_axis=seq_axis,
        name=name + ".reduce_scatter"
    )
    return tensor_replace(z_seq_shard, tp_group=tp_group, shard=ShardSpec("sharded", axis=seq_axis, parts=tp_group.size))


def megatron_mlp_sp(
    x: Tensor, 
    w1_col: Tensor, 
    w2_row: Tensor, 
    *, 
    tp_group: Group, 
    seq_axis: int = 0,
    name: str = "mlp_sp"
) -> Tensor:
    """
    Megatron-style MLP with Tensor Parallelism + Sequence Parallelism.
    
    Communication pattern:
    - AllGather before column-parallel linear (gather full sequence)
    - ReduceScatter after row-parallel linear (sum + scatter to sequence-sharded)
    
    This is more memory efficient than plain TP because activations between
    the two linears have full sequence but are computed locally, while
    input/output activations are sequence-sharded.
    
    Args:
        x: Input tensor, sharded along seq_axis (logical shape: [seq, hidden])
        w1_col: Column-parallel weight (logical shape: [hidden, intermediate])
        w2_row: Row-parallel weight (logical shape: [intermediate, hidden])
        tp_group: Tensor parallel group
        seq_axis: Axis along which sequence is sharded (default: 0)
        name: Operation name prefix
    
    Returns:
        Output tensor, sharded along seq_axis [seq/TP, hidden]
    """
    # AllGather + column-parallel matmul
    h = sp_column_linear(x, w1_col, tp_group=tp_group, seq_axis=seq_axis, name=name + ".fc1")
    
    # Activation (on full sequence, sharded hidden)
    h = activation(h, name=name + ".act")
    
    # Row-parallel matmul + ReduceScatter
    y = sp_row_linear(h, w2_row, tp_group=tp_group, seq_axis=seq_axis, name=name + ".fc2")
    
    return y