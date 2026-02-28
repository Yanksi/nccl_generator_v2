"""
Test for gradient checkpointing functionality.
"""
from simple_sim import (
    Tensor,
    input_tensor,
    parameter,
    matmul,
    elementwise_unary,
    elementwise_binary,
    backward,
    checkpoint,
    checkpoint_start,
    checkpoint_end,
    clear_checkpoint_registry,
    get_graph,
    print_graph_summary,
)
from simple_sim.visualize import visualize_graph


def test_simple_checkpoint():
    """Test a simple checkpoint region."""
    # Clear registry before test
    clear_checkpoint_registry()
    
    # Build a simple computation graph with checkpoint
    x = input_tensor((4, 8), name="x")
    w1 = parameter((8, 16), name="w1")
    w2 = parameter((16, 8), name="w2")
    
    # Create checkpoint region
    with checkpoint(name="layer1") as ckpt:
        h = ckpt.enter(x)
        h = matmul(h, w1, name="matmul1")
        h = elementwise_unary(h, name="act1")
        h = ckpt.exit(h)
    
    out = matmul(h, w2, name="output")
    
    # Run backward
    grads = backward(out, wrt=[w1, w2])
    
    print("=== Test: Simple Checkpoint ===")
    print(f"Gradient for w1: {grads.get(w1)}")
    print(f"Gradient for w2: {grads.get(w2)}")
    print()
    
    # Visualize
    g = get_graph(out)
    print_graph_summary(g)
    print()
    
    # Check that checkpoint ops are in the graph
    checkpoint_ops = [n for n in g.nodes if n.kind in ("checkpoint_start", "checkpoint_end")]
    print(f"Checkpoint ops found: {len(checkpoint_ops)}")
    for op in checkpoint_ops:
        print(f"  - {op.kind}: id={op.checkpoint_id}")
    
    return len(checkpoint_ops) == 2


def test_nested_checkpoint():
    """Test nested checkpoint regions."""
    clear_checkpoint_registry()
    
    x = input_tensor((4, 8), name="x")
    w1 = parameter((8, 16), name="w1")
    w2 = parameter((16, 8), name="w2")
    w3 = parameter((8, 4), name="w3")
    
    # Outer checkpoint
    with checkpoint(name="outer") as outer:
        h = outer.enter(x)
        
        # Inner checkpoint
        with checkpoint(name="inner") as inner:
            h = inner.enter(h)
            h = matmul(h, w1, name="matmul1")
            h = elementwise_unary(h, name="act1")
            h = inner.exit(h)
        
        h = matmul(h, w2, name="matmul2")
        h = outer.exit(h)
    
    out = matmul(h, w3, name="output")
    
    # Run backward
    grads = backward(out, wrt=[w1, w2, w3])
    
    print("\n=== Test: Nested Checkpoint ===")
    print(f"Gradient for w1: {grads.get(w1)}")
    print(f"Gradient for w2: {grads.get(w2)}")
    print(f"Gradient for w3: {grads.get(w3)}")
    print()
    
    # Visualize
    g = get_graph(out)
    print_graph_summary(g)
    print()
    
    # Check checkpoint ops
    checkpoint_ops = [n for n in g.nodes if n.kind in ("checkpoint_start", "checkpoint_end")]
    print(f"Checkpoint ops found: {len(checkpoint_ops)}")
    for op in checkpoint_ops:
        print(f"  - {op.kind}: id={op.checkpoint_id}")
    
    return len(checkpoint_ops) == 4  # 2 starts + 2 ends


def test_no_checkpoint():
    """Test that graph works without checkpoint (baseline)."""
    clear_checkpoint_registry()
    
    x = input_tensor((4, 8), name="x")
    w1 = parameter((8, 16), name="w1")
    w2 = parameter((16, 8), name="w2")
    
    # No checkpoint
    h = matmul(x, w1, name="matmul1")
    h = elementwise_unary(h, name="act1")
    out = matmul(h, w2, name="output")
    
    # Run backward
    grads = backward(out, wrt=[w1, w2])
    
    print("\n=== Test: No Checkpoint (Baseline) ===")
    print(f"Gradient for w1: {grads.get(w1)}")
    print(f"Gradient for w2: {grads.get(w2)}")
    
    # Check no checkpoint ops
    g = get_graph(out)
    checkpoint_ops = [n for n in g.nodes if n.kind in ("checkpoint_start", "checkpoint_end")]
    print(f"Checkpoint ops found: {len(checkpoint_ops)}")
    
    return len(checkpoint_ops) == 0


def test_checkpoint_with_multiple_ops():
    """Test checkpoint with multiple ops inside."""
    clear_checkpoint_registry()
    
    x = input_tensor((2, 4), name="x")
    w1 = parameter((4, 8), name="w1")
    w2 = parameter((8, 8), name="w2")
    w3 = parameter((8, 4), name="w3")
    
    with checkpoint(name="layers") as ckpt:
        h = ckpt.enter(x)
        h = matmul(h, w1, name="matmul1")
        h = elementwise_unary(h, name="act1")
        h = matmul(h, w2, name="matmul2")
        h = elementwise_binary(h, elementwise_unary(h, name="act2"), name="add")
        h = ckpt.exit(h)
    
    out = matmul(h, w3, name="output")
    
    # Run backward
    grads = backward(out, wrt=[w1, w2, w3])
    
    print("\n=== Test: Multiple Ops in Checkpoint ===")
    g = get_graph(out)
    print_graph_summary(g)
    print()
    
    # Count checkpoint and compute ops
    checkpoint_ops = [n for n in g.nodes if n.kind in ("checkpoint_start", "checkpoint_end")]
    compute_ops = [n for n in g.nodes if hasattr(n, 'kind') and n.kind in ("matmul", "elementwise")]
    
    print(f"Checkpoint ops: {len(checkpoint_ops)}")
    print(f"Compute ops in graph: {len(compute_ops)}")
    
    return len(checkpoint_ops) == 2


if __name__ == "__main__":
    results = []
    
    results.append(("No checkpoint (baseline)", test_no_checkpoint()))
    results.append(("Simple checkpoint", test_simple_checkpoint()))
    results.append(("Nested checkpoint", test_nested_checkpoint()))
    results.append(("Multiple ops in checkpoint", test_checkpoint_with_multiple_ops()))
    
    print("\n" + "=" * 50)
    print("TEST RESULTS:")
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(r[1] for r in results)
    print("=" * 50)
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
