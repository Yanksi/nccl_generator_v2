"""Compute operations — ops that model real arithmetic / data movement.

Contents:
- Tensor factories: input_tensor, parameter
- ElementwiseOp   (+ ones_like, activation, add, …)
- ShapeChangeOp   (reshape, transpose, reduction)
- MatMulOp        (matmul)
- AttentionOp / AttentionBwdOp (attention)
- adam_update / adam_update_shard
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence

from .ir import ComputeOp, CostMeta, Group, MemoryCategory, ShardSpec, Tensor, tensor_replace
from .utils import numel


# -------- tensor factories --------

def input_tensor(shape: Sequence[int], *, dtype: str = "fp16", name: str | None = None) -> Tensor:
    return Tensor(tuple(shape), dtype=dtype, producer=None, requires_grad=False, name=name)

def parameter(
    shape: Sequence[int],
    *,
    dtype: str = "fp16",
    name: str | None = None,
    dp_group: Group | None = None,
    tp_group: Group | None = None,
    shard: ShardSpec | None = None,
) -> Tensor:
    return Tensor(
        tuple(shape),
        dtype=dtype,
        producer=None,
        requires_grad=True,
        name=name,
        dp_group=dp_group,
        tp_group=tp_group,
        shard=shard or ShardSpec("replicated"),
    )


# -------- ElementwiseOp --------

@dataclass(frozen=True, eq=False)
class ElementwiseOp(ComputeOp):
    """
    Generic elementwise operation on N input tensors producing M output tensors.
    
    All elementwise ops are memory-bound on GPU, so we model memory transfer only.
    FLOPs are approximate and don't affect scheduling significantly.
    
    Examples:
    - Unary (activation, exp, log): num_inputs=1, num_outputs=1
    - Binary (add, mul, sub): num_inputs=2, num_outputs=1
    - Adam update: num_inputs=4, num_outputs=3 (reads p,g,m,v; writes p,m,v)
    """
    n: int = 0  # elements per tensor
    num_inputs: int = 1
    num_outputs: int = 1
    kind: str = "elementwise"

    def get_cost_meta(self) -> CostMeta:
        return {
            "flops": self.n,  # Approximate, doesn't matter for memory-bound
            "mem_read": self.num_inputs * self.n,
            "mem_write": self.num_outputs * self.n,
        }

    def vjp(self, grad_output: Tensor) -> Dict[Tensor, Tensor]:
        # Backward for elementwise ops is also elementwise with similar cost per input
        tensors = [v for v in self.inputs if isinstance(v, Tensor)]
        grads: Dict[Tensor, Tensor] = {}
        for t in tensors:
            if t.requires_grad:
                grads[t] = elementwise(grad_output, name=f"d({t.name})")
        return grads


def elementwise(*inputs: Tensor, num_outputs: int = 1, kind: str = "elementwise", name: str | None = None) -> Tensor:
    """
    Generic elementwise operation on N tensors.
    
    All input tensors must have the same shape. Returns a single output tensor.
    For ops with multiple outputs (like Adam), use num_outputs > 1.
    """
    assert len(inputs) >= 1, "elementwise requires at least one input"
    shape = inputs[0].shape
    for t in inputs[1:]:
        assert t.shape == shape, f"shape mismatch: {shape} vs {t.shape}"
    
    # Use physical shape for cost model (actual per-rank computation)
    phys_n = numel(inputs[0].physical_shape())
    node = ElementwiseOp(n=phys_n, num_inputs=len(inputs), num_outputs=num_outputs, kind=kind, inputs=inputs)
    # Propagate shard spec from first input
    first_shard = inputs[0].shard
    return Tensor(
        shape=shape,
        dtype=inputs[0].dtype,
        producer=node,
        requires_grad=any(t.requires_grad for t in inputs),
        name=name,
        tp_group=next((t.tp_group for t in inputs if t.tp_group), None),
        dp_group=next((t.dp_group for t in inputs if t.dp_group), None),
        shard=first_shard,
    )


def ones_like(x: Tensor, *, name: str | None = None) -> Tensor:
    """Create a tensor of ones with the same shape (no gradient needed)."""
    node = ElementwiseOp(n=numel(x.physical_shape()), num_inputs=1, num_outputs=1, kind="ones_like", inputs=(x,))
    return Tensor(shape=x.shape, dtype=x.dtype, producer=node, requires_grad=False,
                  name=name, tp_group=x.tp_group, dp_group=x.dp_group,
                  memory_category=MemoryCategory.NOT_MATERIALIZED,
                  shard=x.shard)


# Convenience aliases for backward compatibility
def elementwise_unary(x: Tensor, *, name: str | None = None) -> Tensor:
    return elementwise(x, kind="elementwise_unary", name=name)

def elementwise_binary(a: Tensor, b: Tensor, *, name: str | None = None) -> Tensor:
    return elementwise(a, b, kind="elementwise_binary", name=name)

def activation(x: Tensor, *, name: str | None = None) -> Tensor:
    return elementwise(x, kind="activation", name=name)

def add(a: Tensor, b: Tensor, *, name: str | None = None) -> Tensor:
    return elementwise(a, b, kind="add", name=name)


# -------- ShapeChangeOp --------

@dataclass(frozen=True, eq=False)
class ShapeChangeOp(ComputeOp):
    """
    Operation that changes tensor shape with configurable cost model.
    
    Covers:
    - View ops (reshape, transpose): has_cost=False, zero data movement
    - Reduction (sum, mean): has_cost=True, reads input_n, writes output_n
    - Broadcast: has_cost=True, reads input_n, writes output_n (opposite direction)
    
    The VJP is always a shape change in the opposite direction with the same cost model.
    """
    input_n: int = 0
    output_n: int = 0
    has_cost: bool = False  # If True, actual data movement; if False, zero-cost view
    kind: str = "reshape"   # Configurable: "reshape", "reduction", "broadcast"

    def get_cost_meta(self) -> CostMeta:
        if self.has_cost:
            return {"flops": self.output_n, "mem_read": self.input_n, "mem_write": self.output_n}
        return {"flops": 0, "mem_read": 0, "mem_write": 0}

    def vjp(self, grad_output: Tensor) -> Dict[Tensor, Tensor]:
        (x,) = [v for v in self.inputs if isinstance(v, Tensor)]
        if not x.requires_grad:
            return {}
        # Backward is shape change in opposite direction
        bwd_kind = "broadcast" if self.kind == "reduction" else ("reduction" if self.kind == "broadcast" else "reshape")
        bwd_node = ShapeChangeOp(
            input_n=self.output_n,
            output_n=self.input_n,
            has_cost=self.has_cost,
            kind=bwd_kind,
            inputs=(grad_output,),
        )
        mem_cat = MemoryCategory.NOT_MATERIALIZED if not self.has_cost else MemoryCategory.MATERIALIZED
        alias = grad_output if not self.has_cost else None
        return {x: tensor_replace(x, producer=bwd_node, name=f"d({x.name})", memory_category=mem_cat, aliases=alias)}


def reshape(x: Tensor, *, shape: Sequence[int], shard: ShardSpec | None = None, name: str | None = None) -> Tensor:
    """Reshape tensor to new shape. Zero-cost view.

    If ``shard`` is provided the reshape is shard-aware: the logical
    numels of input and output are compared, and the given shard spec
    is propagated to the output.  Otherwise the physical numels must
    match and the shard spec is cleared (safe default for reshapes
    that may break the shard-axis relationship).
    """
    if shard is not None and shard.kind == "sharded":
        # Shard-aware reshape: compare logical numels
        log_in = numel(x.shape)
        log_out = numel(tuple(shape))
        assert log_in == log_out, (
            f"reshape logical size mismatch: {x.shape} ({log_in}) -> {shape} ({log_out})"
        )
        phys_n = numel(x.physical_shape())
        out_shard = shard
    else:
        phys_in = numel(x.physical_shape())
        phys_out = numel(tuple(shape))
        assert phys_in == phys_out, f"reshape size mismatch: physical {x.physical_shape()} ({phys_in}) -> {shape} ({phys_out})"
        phys_n = phys_in
        out_shard = ShardSpec("replicated")

    node = ShapeChangeOp(input_n=phys_n, output_n=phys_n, has_cost=False, kind="reshape", inputs=(x,))
    return Tensor(shape=tuple(shape), dtype=x.dtype, producer=node, requires_grad=x.requires_grad,
                  name=name, tp_group=x.tp_group, dp_group=x.dp_group,
                  shard=out_shard,
                  memory_category=MemoryCategory.NOT_MATERIALIZED, aliases=x)


def transpose(x: Tensor, *, name: str | None = None) -> Tensor:
    """Transpose a 2D tensor. Zero-cost view.

    If the tensor is sharded, the shard axis is swapped (0↔1).
    """
    assert len(x.shape) == 2, f"transpose requires 2D tensor, got {x.shape}"
    new_shape = (x.shape[1], x.shape[0])
    # Swap shard axis for 2D transpose
    if x.shard.kind == "sharded" and x.shard.axis is not None:
        norm_axis = x.shard.axis % 2  # normalize for 2D
        new_axis = 1 - norm_axis
        new_shard = ShardSpec("sharded", axis=new_axis, parts=x.shard.parts)
    else:
        new_shard = x.shard
    phys_n = numel(x.physical_shape())
    node = ShapeChangeOp(input_n=phys_n, output_n=phys_n, has_cost=False, kind="reshape", inputs=(x,))
    return Tensor(shape=new_shape, dtype=x.dtype, producer=node, requires_grad=x.requires_grad,
                  name=name, tp_group=x.tp_group, dp_group=x.dp_group,
                  memory_category=MemoryCategory.NOT_MATERIALIZED, aliases=x,
                  shard=new_shard)


def reduction(x: Tensor, *, output_shape: Sequence[int], name: str | None = None) -> Tensor:
    """Reduction operation (sum, mean, max, etc.)."""
    phys_in = numel(x.physical_shape())
    phys_out = numel(tuple(output_shape))
    node = ShapeChangeOp(input_n=phys_in, output_n=phys_out, has_cost=True, kind="reduction", inputs=(x,))
    return Tensor(shape=tuple(output_shape), dtype=x.dtype, producer=node, requires_grad=x.requires_grad,
                  name=name, tp_group=x.tp_group, dp_group=x.dp_group)


# -------- MatMulOp --------

@dataclass(frozen=True, eq=False)
class MatMulOp(ComputeOp):
    """
    Matrix multiplication: C[m,n] = A[m,k] @ B[k,n]
    
    FLOPs: 2*m*n*k (multiply-add per output element)
    Memory: Read A (m*k) + B (k*n), Write C (m*n)
    
    This is typically compute-bound on modern GPUs.
    """
    m: int = 0
    n: int = 0
    k: int = 0
    kind: str = field(default="matmul", init=False)

    def get_cost_meta(self) -> CostMeta:
        return {
            "flops": 2 * self.m * self.n * self.k,
            "mem_read": self.m * self.k + self.k * self.n,
            "mem_write": self.m * self.n,
        }

    def vjp(self, grad_output: Tensor) -> Dict[Tensor, Tensor]:
        a, b = [v for v in self.inputs if isinstance(v, Tensor)]
        grads: Dict[Tensor, Tensor] = {}
        if a.requires_grad:
            bT = transpose(b, name=f"{b.name}_T_for_da")
            grads[a] = matmul(grad_output, bT, name=f"d({a.name})")
        if b.requires_grad:
            aT = transpose(a, name=f"{a.name}_T_for_db")
            grads[b] = matmul(aT, grad_output, name=f"d({b.name})")
        return grads

def matmul(a: Tensor, b: Tensor, *, name: str | None = None) -> Tensor:
    assert len(a.shape) == 2 and len(b.shape) == 2
    # Use physical shapes for dimension matching and FLOPs
    a_phys = a.physical_shape()
    b_phys = b.physical_shape()
    m, k1 = a_phys
    k2, n = b_phys
    assert k1 == k2, f"matmul shape mismatch: physical {a_phys} x {b_phys} (logical {a.shape} x {b.shape})"

    # Output logical shape: derive from logical input shapes
    # For A[M_log, K_log] @ B[K_log, N_log] -> C[M_log, N_log]
    # But the physical computation is A_phys @ B_phys = [m, n]
    # The output inherits sharding from the inputs on the output-facing axes.
    out_m_log = a.shape[0]  # rows from A (logical)
    out_n_log = b.shape[1]  # cols from B (logical)
    out_shape = (out_m_log, out_n_log)

    # Infer output shard spec:
    # If A is sharded on axis 0 -> output is sharded on axis 0 (rows)
    # If B is sharded on axis 1 -> output is sharded on axis 1 (cols)
    # If both are sharded on the contracting dim, output is replicated
    out_shard = ShardSpec("replicated")
    if a.shard.kind == "sharded" and a.shard.axis == 0:
        out_shard = a.shard
    elif b.shard.kind == "sharded" and b.shard.axis == 1:
        out_shard = b.shard

    node = MatMulOp(m=m, n=n, k=k1, inputs=(a, b))
    out = Tensor(
        shape=out_shape,
        dtype=a.dtype,
        producer=node,
        requires_grad=(a.requires_grad or b.requires_grad),
        name=name,
        tp_group=a.tp_group or b.tp_group,
        dp_group=a.dp_group or b.dp_group,
        shard=out_shard,
    )
    return out


# -------- AttentionOp --------

@dataclass(frozen=True, eq=False)
class AttentionOp(ComputeOp):
    """
    FlashAttention-style fused attention operation.
    
    Forward pass:
      S = Q @ K^T  -> (batch, heads, seq_q, seq_kv): 2*B*H*Sq*Skv*D FLOPs
      P = softmax(S)  -> ~5*B*H*Sq*Skv FLOPs (exp, sum, div)
      O = P @ V     -> 2*B*H*Sq*Skv*D FLOPs
      Total: ~4*B*H*Sq*Skv*D + 5*B*H*Sq*Skv FLOPs
    
    Backward pass is ~2x forward cost (4 matmuls instead of 2, plus softmax backward).
    
    Memory (FlashAttention): O(B*H*Sq*D) instead of O(B*H*Sq*Skv) - no attention matrix materialized.
    """
    batch_size: int = 0
    seq_len_q: int = 0
    seq_len_kv: int = 0
    num_heads: int = 0
    head_dim: int = 0
    kind: str = field(default="attention", init=False)

    def get_cost_meta(self) -> CostMeta:
        B, H, Sq, Skv, D = (self.batch_size, self.num_heads, 
                            self.seq_len_q, self.seq_len_kv, self.head_dim)
        return {
            "flops": 4 * B * H * Sq * Skv * D,
            "mem_read": B * H * D * (Sq + 2 * Skv),
            "mem_write": B * H * Sq * D,
        }

    def vjp(self, grad_output: Tensor) -> Dict[Tensor, Tensor]:
        q, k, v = [t for t in self.inputs if isinstance(t, Tensor)]
        
        # Create a single backward attention op with 2x cost
        bwd_node = AttentionBwdOp(
            batch_size=self.batch_size,
            seq_len_q=self.seq_len_q,
            seq_len_kv=self.seq_len_kv,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            inputs=(grad_output, q, k, v),
        )
        
        grads: Dict[Tensor, Tensor] = {}
        if q.requires_grad:
            grads[q] = tensor_replace(q, producer=bwd_node, name=f"d({q.name})")
        if k.requires_grad:
            grads[k] = tensor_replace(k, producer=bwd_node, name=f"d({k.name})")
        if v.requires_grad:
            grads[v] = tensor_replace(v, producer=bwd_node, name=f"d({v.name})")

        return grads


@dataclass(frozen=True, eq=False)
class AttentionBwdOp(ComputeOp):
    """
    Backward pass for FlashAttention.
    
    Backward has 4 matmuls instead of 2 in forward, so ~2x forward cost.
    """
    batch_size: int = 0
    seq_len_q: int = 0
    seq_len_kv: int = 0
    num_heads: int = 0
    head_dim: int = 0
    kind: str = field(default="attention_bwd", init=False)

    def get_cost_meta(self) -> CostMeta:
        B, H, Sq, Skv, D = (self.batch_size, self.num_heads, 
                            self.seq_len_q, self.seq_len_kv, self.head_dim)
        return {
            "flops": 2 * 4 * B * H * Sq * Skv * D,  # 2x forward
            "mem_read": B * H * D * (2 * Sq + 2 * Skv),  # dO, Q, K, V
            "mem_write": B * H * D * (Sq + 2 * Skv),  # dQ, dK, dV
        }

    def vjp(self, grad_output: Tensor) -> Dict[Tensor, Tensor]:
        raise NotImplementedError("AttentionBwdOp does not have a vjp")


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    name: str | None = None,
) -> Tensor:
    """
    FlashAttention-style fused attention.
    
    Args:
        q: Query tensor of shape (batch, seq_q, num_heads, head_dim)
        k: Key tensor of shape (batch, seq_kv, num_heads, head_dim)
        v: Value tensor of shape (batch, seq_kv, num_heads, head_dim)
    
    Returns:
        Output tensor of shape (batch, seq_q, num_heads, head_dim)
    """
    assert len(q.shape) == 4, f"Expected 4D tensor for Q, got {q.shape}"
    assert len(k.shape) == 4, f"Expected 4D tensor for K, got {k.shape}"
    assert len(v.shape) == 4, f"Expected 4D tensor for V, got {v.shape}"
    
    # Use physical shapes for dimension matching and cost model
    q_phys = q.physical_shape()
    k_phys = k.physical_shape()
    v_phys = v.physical_shape()
    
    batch, seq_q, num_heads, head_dim = q_phys
    batch_k, seq_kv, num_heads_k, head_dim_k = k_phys
    batch_v, seq_kv_v, num_heads_v, head_dim_v = v_phys
    
    assert batch == batch_k == batch_v, "Batch size mismatch"
    assert num_heads == num_heads_k == num_heads_v, "Number of heads mismatch"
    assert head_dim == head_dim_k == head_dim_v, "Head dimension mismatch"
    assert seq_kv == seq_kv_v, "K and V sequence length mismatch"

    node = AttentionOp(
        batch_size=batch,
        seq_len_q=seq_q,
        seq_len_kv=seq_kv,
        num_heads=num_heads,
        head_dim=head_dim,
        inputs=(q, k, v),
    )
    out = Tensor(
        shape=q.shape,  # Output has same logical shape as Q
        dtype=q.dtype,
        producer=node,
        requires_grad=(q.requires_grad or k.requires_grad or v.requires_grad),
        name=name,
        tp_group=q.tp_group or k.tp_group or v.tp_group,
        dp_group=q.dp_group or k.dp_group or v.dp_group,
        shard=q.shard,
    )
    return out


# -------- Optimizer ops --------

def adam_update_shard(param_full: Tensor, grad_shard: Tensor, *, name: str | None = None) -> Tensor:
    """
    Adam optimizer update on a parameter shard (ZeRO-1 style).
    
    Models the memory-bound Adam update: reads param, grad, momentum, variance (4 tensors),
    writes updated param, momentum, variance (3 tensors).
    
    Args:
        param_full: Parameter tensor (used for metadata; in ZeRO-1, this is sharded)
        grad_shard: Gradient shard to apply
    
    Returns:
        Updated parameter shard
    """
    node = ElementwiseOp(
        n=numel(grad_shard.physical_shape()),
        num_inputs=4,  # reads: param, grad, momentum, variance
        num_outputs=3,  # writes: param, momentum, variance
        kind="adam_update",
        inputs=(param_full, grad_shard),
    )
    return Tensor(
        shape=grad_shard.shape,
        dtype=param_full.dtype,
        producer=node,
        requires_grad=False,
        name=name,
        dp_group=param_full.dp_group,
        tp_group=param_full.tp_group,
        shard=grad_shard.shard,
    )


def adam_update(param: Tensor, grad: Tensor, *, name: str | None = None) -> Tensor:
    """
    Simple Adam optimizer update (non-distributed).
    
    Models the memory-bound Adam update: reads param, grad, momentum, variance (4 tensors),
    writes updated param, momentum, variance (3 tensors).
    
    This is a simplified version without distributed training overhead.
    
    Args:
        param: Parameter tensor to update
        grad: Gradient tensor
    
    Returns:
        Updated parameter tensor
    """
    node = ElementwiseOp(
        n=numel(grad.physical_shape()),
        num_inputs=4,  # reads: param, grad, momentum, variance
        num_outputs=3,  # writes: param, momentum, variance
        kind="adam_update",
        inputs=(param, grad),
    )
    return Tensor(
        shape=grad.shape,
        dtype=param.dtype,
        producer=node,
        requires_grad=False,  # Optimizer output doesn't need gradients
        name=name,
        dp_group=param.dp_group,
        tp_group=param.tp_group,
        shard=param.shard,
    )
