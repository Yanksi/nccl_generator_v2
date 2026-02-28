"""
Llama3 Training Simulation Script

Cluster configuration:
- 32 GPUs total (8 nodes × 4 GPUs/node)
- TP=4 (tensor parallelism within a node)
- DP=4 (data parallelism across nodes)  
- PP=2 (pipeline parallelism, 2 stages)

This script generates the computation graph for 2 training iterations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict

from simple_sim import (
    Group, ShardSpec, Tensor, Token, MemoryCategory,
    input_tensor, parameter,
    fill, send, sink,
    matmul, attention, elementwise_unary, elementwise_binary, reduction, reshape,
    detach, wait_for, param_factory,
    allreduce,
    backward,
    Zero1Plan, zero1_optimizer_step,
    get_graph, topo_sort, ExtractedGraph,
)
from simple_sim.tp_megatron import tp_column_linear, tp_row_linear


# ============================================================================
# Llama3 Model Configuration
# ============================================================================

@dataclass
class Llama3Config:
    """Llama3 8B-like configuration"""
    hidden_size: int = 4096
    intermediate_size: int = 14336  # FFN hidden dim (3.5x hidden for SwiGLU)
    num_attention_heads: int = 32
    num_kv_heads: int = 8           # GQA: grouped query attention
    head_dim: int = 128             # hidden_size // num_attention_heads
    num_layers: int = 32
    vocab_size: int = 128256
    max_seq_len: int = 8192
    
    # Training config
    batch_size: int = 4             # micro-batch size per DP rank
    seq_len: int = 2048             # sequence length for this run


# ============================================================================
# Llama3 Transformer Block (with TP)
# ============================================================================

def llama3_attention(
    x: Tensor,
    wq: Tensor,
    wk: Tensor,
    wv: Tensor,
    wo: Tensor,
    *,
    tp_group: Group,
    config: Llama3Config,
    name: str,
) -> Tensor:
    """
    Llama3 attention with GQA (Grouped Query Attention).
    
    TP strategy:
    - Q, K, V projections: column parallel (shard heads across TP)
    - Output projection: row parallel (allreduce at the end)
    """
    batch_seq = x.shape[0]  # batch * seq_len (flattened)
    
    # Q, K, V projections (column parallel)
    q = tp_column_linear(x, wq, tp_group=tp_group, name=f"{name}.q_proj")
    k = tp_column_linear(x, wk, tp_group=tp_group, name=f"{name}.k_proj")
    v = tp_column_linear(x, wv, tp_group=tp_group, name=f"{name}.v_proj")
    
    # Reshape for attention: logical (batch*seq, num_heads * head_dim) -> (batch, seq, num_heads, head_dim)
    # Use logical (full) head count; shard spec tracks how heads are split across TP ranks.
    batch = config.batch_size
    seq = config.seq_len
    num_heads = config.num_attention_heads
    head_dim = config.head_dim
    tp_heads_shard = ShardSpec("sharded", axis=2, parts=tp_group.size)
    
    # Reshape Q, K, V to (batch, seq, num_heads, head_dim) for attention
    # NOTE: For simplicity, we model all attention heads with full Q/K/V (non-GQA).
    # In GQA, K/V have fewer heads but FlashAttention handles broadcasting internally.
    # For cost modeling, using full heads gives a reasonable upper bound.
    q_4d = reshape(q, shape=(batch, seq, num_heads, head_dim), shard=tp_heads_shard, name=f"{name}.q_reshape")
    k_4d = reshape(k, shape=(batch, seq, num_heads, head_dim), shard=tp_heads_shard, name=f"{name}.k_reshape")
    v_4d = reshape(v, shape=(batch, seq, num_heads, head_dim), shard=tp_heads_shard, name=f"{name}.v_reshape")
    
    # Attention (flash attention style)
    attn_out = attention(q_4d, k_4d, v_4d, name=f"{name}.flash_attn")
    
    # Reshape back to (batch*seq, num_heads * head_dim) — logical shape
    attn_flat = reshape(
        attn_out, 
        shape=(batch * seq, num_heads * head_dim),
        shard=ShardSpec("sharded", axis=1, parts=tp_group.size),
        name=f"{name}.attn_reshape",
    )
    
    # Output projection (row parallel with allreduce)
    out = tp_row_linear(attn_flat, wo, tp_group=tp_group, name=f"{name}.o_proj")
    
    return out


def llama3_mlp(
    x: Tensor,
    w_gate: Tensor,
    w_up: Tensor,
    w_down: Tensor,
    *,
    tp_group: Group,
    name: str,
) -> Tensor:
    """
    Llama3 SwiGLU MLP.
    
    SwiGLU: output = down_proj(silu(gate_proj(x)) * up_proj(x))
    
    TP strategy:
    - gate_proj, up_proj: column parallel
    - down_proj: row parallel (allreduce at the end)
    """
    # Gate and up projections (column parallel, no allreduce)
    gate = tp_column_linear(x, w_gate, tp_group=tp_group, name=f"{name}.gate_proj")
    up = tp_column_linear(x, w_up, tp_group=tp_group, name=f"{name}.up_proj")
    
    # SiLU activation on gate
    gate_act = elementwise_unary(gate, name=f"{name}.silu")
    
    # Element-wise multiply: silu(gate) * up
    hidden = elementwise_binary(gate_act, up, name=f"{name}.gate_mul_up")
    
    # Down projection (row parallel with allreduce)
    out = tp_row_linear(hidden, w_down, tp_group=tp_group, name=f"{name}.down_proj")
    
    return out


def llama3_block(
    x: Tensor,
    attn_params: Dict[str, Tensor],
    mlp_params: Dict[str, Tensor],
    *,
    tp_group: Group,
    config: Llama3Config,
    name: str,
) -> Tensor:
    """
    Single Llama3 transformer block.
    
    Structure:
    - RMSNorm -> Attention -> Residual
    - RMSNorm -> MLP -> Residual
    """
    # Pre-attention RMSNorm (elementwise, memory-bound)
    x_norm1 = elementwise_unary(x, name=f"{name}.input_norm")
    
    # Self-attention
    attn_out = llama3_attention(
        x_norm1,
        attn_params["wq"],
        attn_params["wk"],
        attn_params["wv"],
        attn_params["wo"],
        tp_group=tp_group,
        config=config,
        name=f"{name}.attn",
    )
    
    # Residual connection
    x = elementwise_binary(x, attn_out, name=f"{name}.attn_residual")
    
    # Pre-MLP RMSNorm
    x_norm2 = elementwise_unary(x, name=f"{name}.post_attn_norm")
    
    # MLP
    mlp_out = llama3_mlp(
        x_norm2,
        mlp_params["w_gate"],
        mlp_params["w_up"],
        mlp_params["w_down"],
        tp_group=tp_group,
        name=f"{name}.mlp",
    )
    
    # Residual connection
    x = elementwise_binary(x, mlp_out, name=f"{name}.mlp_residual")
    
    return x


# ============================================================================
# Pipeline Stage Builder
# ============================================================================

def create_layer_params(
    layer_idx: int,
    *,
    tp_group: Group,
    dp_group: Group,
    config: Llama3Config,
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], List[Tensor]]:
    """Create parameters for a single transformer layer."""
    tp_size = tp_group.size
    H = config.hidden_size
    I = config.intermediate_size
    
    num_heads = config.num_attention_heads
    head_dim = config.head_dim
    
    # Attention params — full logical shapes with shard specs
    attn_params = {
        "wq": parameter(
            (H, num_heads * head_dim),
            name=f"layer{layer_idx}.wq",
            tp_group=tp_group,
            dp_group=dp_group,
            shard=ShardSpec("sharded", axis=1, parts=tp_size),
        ),
        "wk": parameter(
            (H, num_heads * head_dim),
            name=f"layer{layer_idx}.wk",
            tp_group=tp_group,
            dp_group=dp_group,
            shard=ShardSpec("sharded", axis=1, parts=tp_size),
        ),
        "wv": parameter(
            (H, num_heads * head_dim),
            name=f"layer{layer_idx}.wv",
            tp_group=tp_group,
            dp_group=dp_group,
            shard=ShardSpec("sharded", axis=1, parts=tp_size),
        ),
        "wo": parameter(
            (num_heads * head_dim, H),
            name=f"layer{layer_idx}.wo",
            tp_group=tp_group,
            dp_group=dp_group,
            shard=ShardSpec("sharded", axis=0, parts=tp_size),
        ),
    }
    
    # MLP params — full logical shapes with shard specs
    mlp_params = {
        "w_gate": parameter(
            (H, I),
            name=f"layer{layer_idx}.w_gate",
            tp_group=tp_group,
            dp_group=dp_group,
            shard=ShardSpec("sharded", axis=1, parts=tp_size),
        ),
        "w_up": parameter(
            (H, I),
            name=f"layer{layer_idx}.w_up",
            tp_group=tp_group,
            dp_group=dp_group,
            shard=ShardSpec("sharded", axis=1, parts=tp_size),
        ),
        "w_down": parameter(
            (I, H),
            name=f"layer{layer_idx}.w_down",
            tp_group=tp_group,
            dp_group=dp_group,
            shard=ShardSpec("sharded", axis=0, parts=tp_size),
        ),
    }
    
    all_params = list(attn_params.values()) + list(mlp_params.values())
    
    return attn_params, mlp_params, all_params


def build_pp_stage(
    *,
    pp_rank: int,
    pp_size: int,
    tp_group: Group,
    dp_group: Group,
    config: Llama3Config,
    iteration: int,
    prev_params: List[Tensor],
) -> Tuple[Tensor, Token, List[Tensor]]:
    """
    Build computation graph for one pipeline stage.
    
    Args:
        prev_params: Parameters to use (provided by param_factory).
    
    Returns: (output_tensor, sync_token, all_parameters)
    """
    layers_per_stage = config.num_layers // pp_size
    first_layer = pp_rank * layers_per_stage
    last_layer = first_layer + layers_per_stage
    
    batch_seq = config.batch_size * config.seq_len
    H = config.hidden_size
    
    pp_prev = pp_rank - 1 if pp_rank > 0 else None
    pp_next = pp_rank + 1 if pp_rank < pp_size - 1 else None
    
    placeholder = None

    # Pipeline input
    if pp_prev is None:
        # First stage: input embedding (simplified as input tensor)
        x = input_tensor((batch_seq, H), name=f"iter{iteration}.embed")
    else:
        # Receive activation from previous stage
        placeholder = Tensor(
            shape=(batch_seq, H), dtype="fp16",
            memory_category=MemoryCategory.NOT_MATERIALIZED,
            requires_grad=True,
            name=f"iter{iteration}.pp_recv.placeholder",
        )
        x = fill(
            placeholder, src=pp_prev,
            tag=iteration * 1000 + pp_rank,
            name=f"iter{iteration}.pp_recv",
        )
    
    all_params: List[Tensor] = []
    param_idx = 0
    
    # Process layers in this stage
    for layer_idx in range(first_layer, last_layer):
        # Each layer has 7 params: wq, wk, wv, wo, w_gate, w_up, w_down
        layer_params = prev_params[param_idx:param_idx + 7]
        param_idx += 7
        attn_params = {
            "wq": layer_params[0],
            "wk": layer_params[1],
            "wv": layer_params[2],
            "wo": layer_params[3],
        }
        mlp_params = {
            "w_gate": layer_params[4],
            "w_up": layer_params[5],
            "w_down": layer_params[6],
        }
        all_params.extend(layer_params)
        
        x = llama3_block(
            x,
            attn_params,
            mlp_params,
            tp_group=tp_group,
            config=config,
            name=f"iter{iteration}.layer{layer_idx}",
        )
    
    # Pipeline output
    if pp_next is None:
        # Last stage: compute loss
        # Final RMSNorm
        x = elementwise_unary(x, name=f"iter{iteration}.final_norm")
        
        # Output projection (simplified - in reality this is a large vocab projection)
        # For Llama3, this would be (batch*seq, hidden) -> (batch*seq, vocab_size)
        # We simplify to just a reduction for loss computation
        loss = reduction(x, output_shape=(1,), name=f"iter{iteration}.loss")
    else:
        # Send activation to next stage — send.vjp auto-recvs gradient
        loss = send(
            x,
            dst=pp_next,
            tag=iteration * 1000 + pp_rank + 1,
            name=f"iter{iteration}.pp_send",
        )
    
    return loss, placeholder, all_params


def build_training_iteration(
    *,
    pp_rank: int,
    pp_size: int,
    tp_group: Group,
    dp_group: Group,
    config: Llama3Config,
    iteration: int,
    prev_params: List[Tensor] | None = None,
) -> Tuple[Token, List[Tensor]]:
    """
    Build one complete training iteration (forward + backward + optimizer).
    
    Returns: (done_token, updated_params)
        - done_token: Token marking completion of this iteration
        - updated_params: Raw params from optimizer (NOT detached - caller handles chaining)
    """
    # Forward pass - use prev_params if provided (from previous iteration)
    loss, placeholder, params = build_pp_stage(
        pp_rank=pp_rank,
        pp_size=pp_size,
        tp_group=tp_group,
        dp_group=dp_group,
        config=config,
        iteration=iteration,
        prev_params=prev_params,
    )
    
    # Backward pass — send.vjp auto-recvs grad; fill.vjp auto-sends grad
    wrt = list(params)
    if placeholder is not None:
        wrt.append(placeholder)
    grads = backward(loss, wrt=wrt)
    
    # Collect backward PP send dependency
    deps: list = []
    if placeholder is not None and placeholder in grads:
        deps.append(grads[placeholder])
    step_tok = sink(*deps, name=f"iter{iteration}.pp_bwd_sync") if deps else sink(name=f"iter{iteration}.pp_bwd_sync")
    
    # ZeRO-1 optimizer step
    plan = Zero1Plan(dp_group=dp_group, gather_policy="eager_allgather")
    new_params, opt_tok = zero1_optimizer_step(
        params,
        grads,
        plan=plan,
        after=step_tok,
        name=f"iter{iteration}.zero1",
    )
    
    # Return raw params - let caller decide on chaining policy via param_factory
    return opt_tok, new_params


def build_full_training(
    *,
    pp_rank: int,
    num_iterations: int = 2,
) -> "ExtractedGraph":
    """
    Build computation graph for multiple training iterations.
    
    Cluster config:
    - 32 GPUs: 8 nodes × 4 GPUs/node
    - TP=4 (within node)
    - DP=4 (across nodes)
    - PP=2 (pipeline stages)
    """
    # Parallelism configuration
    tp_size = 4
    dp_size = 4
    pp_size = 2
    
    tp_group = Group("tp", id=0, size=tp_size)
    dp_group = Group("dp", id=0, size=dp_size)
    
    # Model configuration
    config = Llama3Config(
        # NOTE: Using smaller config for faster graph construction
        # Full Llama3 8B would use: hidden=4096, intermediate=14336, 
        # heads=32, layers=32, batch=4, seq=2048
        # But graph construction takes too long with those values.
        hidden_size=1024,
        intermediate_size=2816,  # 2.75x hidden (scaled down)
        num_attention_heads=8,
        num_kv_heads=8,
        head_dim=128,
        num_layers=8,  # 4 layers per PP stage
        batch_size=1,
        seq_len=512,
    )
    
    print(f"Llama3 Training Simulation")
    print(f"=" * 50)
    print(f"Cluster: 32 GPUs (8 nodes × 4 GPUs/node)")
    print(f"Parallelism: TP={tp_size}, DP={dp_size}, PP={pp_size}")
    print(f"Pipeline rank: {pp_rank}")
    print(f"Layers per stage: {config.num_layers // pp_size}")
    print(f"Micro-batch: {config.batch_size} × {config.seq_len} tokens")
    print(f"Iterations: {num_iterations}")
    print(f"=" * 50)
    
    # Create function for fresh params (first iteration)
    def make_fresh_params() -> List[Tensor]:
        layers_per_stage = config.num_layers // pp_size
        first_layer = pp_rank * layers_per_stage
        last_layer = first_layer + layers_per_stage
        all_params = []
        for layer_idx in range(first_layer, last_layer):
            _, _, layer_params = create_layer_params(
                layer_idx,
                tp_group=tp_group,
                dp_group=dp_group,
                config=config,
            )
            all_params.extend(layer_params)
        return all_params
    
    # Build iterations with flexible param chaining
    tokens: List[Token] = []
    raw_params: List[Tensor] | None = None
    prev_tok: Token | None = None
    
    for it in range(num_iterations):
        # Use param_factory to prepare params for this iteration
        # Option 1: after=None -> params available immediately after update (default)
        # Option 2: after=prev_tok -> params wait for previous iteration to complete
        iter_params = param_factory(
            raw_params,
            after=prev_tok,  # Wait for previous iteration to complete
            create_fn=make_fresh_params,
            iteration=it,
        )
        
        # Build the iteration
        tok, raw_params = build_training_iteration(
            pp_rank=pp_rank,
            pp_size=pp_size,
            tp_group=tp_group,
            dp_group=dp_group,
            config=config,
            iteration=it,
            prev_params=iter_params,
        )
        tokens.append(tok)
        prev_tok = tok
    
    # Combine all iteration tokens
    final_tok = sink(*tokens, name="training_done")
    
    # Extract full graph
    graph = get_graph(final_tok)
    
    return graph


def analyze_graph(graph) -> None:
    """Analyze and print statistics about the computation graph."""
    nodes = topo_sort(graph.nodes)
    
    # Count ops by kind
    op_counts: Dict[str, int] = {}
    for n in nodes:
        kind = getattr(n, "kind", type(n).__name__)
        op_counts[kind] = op_counts.get(kind, 0) + 1
    
    print(f"\nGraph Statistics:")
    print(f"-" * 40)
    print(f"Total nodes: {len(nodes)}")
    print(f"\nOp counts by kind:")
    for kind, count in sorted(op_counts.items(), key=lambda x: -x[1]):
        print(f"  {kind}: {count}")
    
    # Count communication ops
    comm_ops = ["allreduce", "reduce_scatter", "allgather", "send", "recv"]
    comm_count = sum(op_counts.get(op, 0) for op in comm_ops)
    print(f"\nTotal communication ops: {comm_count}")
    
    # Count compute-bound ops
    compute_ops = ["matmul", "attention", "attention_bwd"]
    compute_count = sum(op_counts.get(op, 0) for op in compute_ops)
    print(f"Total compute-bound ops: {compute_count}")
    
    # Count memory-bound ops
    memory_ops = ["elementwise_unary", "elementwise_binary", "reduction", "reduction_bwd"]
    memory_count = sum(op_counts.get(op, 0) for op in memory_ops)
    print(f"Total memory-bound ops: {memory_count}")


if __name__ == "__main__":
    # Build graph for PP rank 0 (first pipeline stage)
    print("\n" + "=" * 60)
    print("Building graph for Pipeline Stage 0 (layers 0-15)")
    print("=" * 60)
    graph_stage0 = build_full_training(pp_rank=0, num_iterations=2)
    analyze_graph(graph_stage0)
    
    # Build graph for PP rank 1 (second pipeline stage)  
    print("\n" + "=" * 60)
    print("Building graph for Pipeline Stage 1 (layers 16-31)")
    print("=" * 60)
    graph_stage1 = build_full_training(pp_rank=1, num_iterations=2)
    analyze_graph(graph_stage1)
