"""
Computation graph visualizer using graphviz.

Usage:
    from simple_sim.visualize import visualize_graph, to_dot
    
    # From tensors/tokens
    dot_str = to_dot(loss, grads[x], title="Forward + Backward")
    
    # Render to file
    visualize_graph(loss, output="graph.svg")
    visualize_graph(loss, output="graph.png", format="png")
    
    # From extracted graph
    from simple_sim.extract import get_graph
    g = get_graph(loss)
    visualize_graph(g, output="graph.pdf")
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

from .ir import MemoryCategory, OpNode, Tensor, Token, Value, CommOp, ComputeOp
from .extract import get_graph, ExtractedGraph, get_activation_summary, ActivationMemoryInfo


# Color schemes for different op types
OP_COLORS = {
    # Compute ops
    "matmul": "#4CAF50",        # Green - compute heavy
    "attention": "#8BC34A",     # Light green
    "attention_bwd": "#689F38", # Darker green
    
    # Elementwise ops (memory bound)
    "elementwise": "#2196F3",   # Blue
    "elementwise_unary": "#2196F3",
    "elementwise_binary": "#2196F3",
    "activation": "#2196F3",
    "add": "#2196F3",
    "adam_update": "#03A9F4",   # Light blue
    
    # Shape ops
    "reshape": "#9E9E9E",       # Gray - zero cost
    "reduction": "#FF9800",     # Orange
    "broadcast": "#FFC107",     # Amber
    
    # Control flow
    "detach": "#E91E63",        # Pink - gradient barrier
    "wait_for": "#9C27B0",      # Purple - scheduling
    
    # Gradient checkpointing
    "checkpoint_start": "#FF5722",  # Deep orange
    "checkpoint_end": "#FF5722",   # Deep orange
    
    # Communication
    "allreduce": "#F44336",     # Red
    "reduce_scatter": "#E53935",
    "allgather": "#EF5350",
    "send": "#D32F2F",          # Red
    "recv": "#E57373",          # Light red (matching send)
    "fill": "#CE93D8",          # Light purple (local fill)
    "sink": "#757575",          # Gray
}

DEFAULT_COLOR = "#607D8B"  # Blue gray


def _get_display_kind(node: OpNode) -> str:
    """Return the display kind for a node (e.g. FillOp -> 'recv' or 'fill')."""
    if hasattr(node, 'display_kind'):
        return node.display_kind
    return node.kind


def _get_node_color(node: OpNode) -> str:
    """Get color for a node based on its display kind."""
    return OP_COLORS.get(_get_display_kind(node), DEFAULT_COLOR)


def _get_node_label(node: OpNode) -> str:
    """Generate a label for a node."""
    kind = _get_display_kind(node)
    lines = [f"<b>{kind}</b>"]
    lines.append(f"id={node.id}")
    
    # Add cost info if available
    try:
        cost = node.get_cost_meta()
        if cost["flops"] > 0 or cost["mem_read"] > 0:
            lines.append(f"flops={_format_num(cost['flops'])}")
            lines.append(f"mem={_format_num(cost['mem_read'] + cost['mem_write'])}")
    except Exception:
        pass
    
    # Add op-specific info
    if hasattr(node, 'm') and hasattr(node, 'n') and hasattr(node, 'k'):
        lines.append(f"[{node.m}×{node.k}]×[{node.k}×{node.n}]")
    elif hasattr(node, 'batch_size') and hasattr(node, 'seq_len_q'):
        lines.append(f"B={node.batch_size}, S={node.seq_len_q}")
    elif hasattr(node, 'n') and hasattr(node, 'num_inputs'):
        lines.append(f"n={node.n}, in={node.num_inputs}")
    elif hasattr(node, 'bytes'):
        lines.append(f"bytes={_format_num(node.bytes)}")

    # Collectives: show parallelism group
    if hasattr(node, 'group'):
        g = node.group
        lines.append(f"<i>{g.kind.upper()} (size={g.size})</i>")

    # PP send/recv: show label, src/dst, and tag
    if hasattr(node, 'label') and node.label:
        lines.append(f"<i>{node.label}</i>")
    if hasattr(node, 'src'):
        lines.append(f"src={node.src}, tag={node.tag}")
    elif hasattr(node, 'dst'):
        lines.append(f"dst={node.dst}, tag={node.tag}")
    
    return "<br/>".join(lines)


def _format_num(n: int) -> str:
    """Format large numbers with K/M/G suffixes."""
    if n >= 1e9:
        return f"{n/1e9:.1f}G"
    elif n >= 1e6:
        return f"{n/1e6:.1f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)


def _get_value_label(v: Value) -> str:
    """Generate label for a tensor/token."""
    if isinstance(v, Tensor):
        name = v.name or "tensor"
        shape_str = "×".join(str(d) for d in v.shape)
        shard_info = ""
        if v.shard.kind == "sharded" and v.shard.axis is not None:
            shard_info = f" (shard ax={v.shard.axis}, ×{v.shard.parts})"
        return f"{name}\\n[{shape_str}]{shard_info}"
    else:
        return v.name or "token"


def to_dot(
    *roots: Union[Value, ExtractedGraph],
    title: Optional[str] = None,
    show_tensors: bool = True,
    show_costs: bool = True,
    rankdir: str = "TB",  # TB (top-bottom) or LR (left-right)
) -> str:
    """
    Generate DOT format string for the computation graph.
    
    Args:
        roots: Tensors, Tokens, or an ExtractedGraph to visualize
        title: Optional title for the graph
        show_tensors: If True, show tensor nodes; if False, only show ops
        show_costs: If True, show cost metadata on nodes
        rankdir: Graph direction - "TB" (top to bottom) or "LR" (left to right)
    
    Returns:
        DOT format string
    """
    # Handle ExtractedGraph input
    if len(roots) == 1 and isinstance(roots[0], ExtractedGraph):
        graph = roots[0]
    else:
        values = [r for r in roots if isinstance(r, (Tensor, Token))]
        graph = get_graph(*values)
    
    nodes = graph.nodes
    
    # Build node ID mapping
    node_ids: Dict[int, str] = {n.id: f"op_{n.id}" for n in nodes}
    
    # Collect all tensors and tokens
    value_ids: Dict[int, str] = {}
    value_objs: Dict[int, Value] = {}
    
    for node in nodes:
        for i, v in enumerate(node.inputs):
            vid = id(v)
            if vid not in value_ids:
                value_ids[vid] = f"val_{vid}"
                value_objs[vid] = v
        for i, v in enumerate(node.outputs):
            vid = id(v)
            if vid not in value_ids:
                value_ids[vid] = f"val_{vid}"
                value_objs[vid] = v
    
    # Also add outputs from tensors that point to these nodes
    for node in nodes:
        # Find tensors that have this node as producer
        pass  # We track via inputs only for simplicity
    
    # Build DOT string
    lines = ["digraph G {"]
    lines.append(f'    rankdir={rankdir};')
    lines.append('    node [fontname="Helvetica", fontsize=10];')
    lines.append('    edge [fontname="Helvetica", fontsize=9];')
    
    if title:
        lines.append(f'    labelloc="t";')
        lines.append(f'    label="{title}";')
        lines.append('    fontsize=14;')
    
    # Add op nodes
    lines.append("")
    lines.append("    // Op nodes")
    for node in nodes:
        nid = node_ids[node.id]
        color = _get_node_color(node)
        label = _get_node_label(node) if show_costs else f"<b>{node.kind}</b><br/>id={node.id}"
        
        # Determine shape based on op type
        if isinstance(node, CommOp):
            shape = "hexagon"
        elif node.kind in ("detach", "wait_for"):
            shape = "diamond"
        else:
            shape = "box"
        
        lines.append(
            f'    {nid} [label=<{label}>, shape={shape}, '
            f'style="filled,rounded", fillcolor="{color}", fontcolor="white"];'
        )
    
    # Add tensor/token nodes if requested
    if show_tensors:
        lines.append("")
        lines.append("    // Value nodes (tensors/tokens)")
        
        # Categorize tensors by type
        for vid, v in value_objs.items():
            val_id = value_ids[vid]
            label = _get_value_label(v)
            
            if isinstance(v, Tensor):
                # Check if this is a graph input (no producer in our extracted graph)
                has_producer_in_graph = (
                    v.producer is not None and v.producer.id in node_ids
                )
                
                if not has_producer_in_graph:
                    if v.requires_grad:
                        # Parameter (trainable weight)
                        fillcolor = '#BBDEFB'  # Light blue
                        border_color = '#1976D2'
                        style = '"filled"'
                        category = "parameter"
                    else:
                        # Input tensor (not trainable)
                        fillcolor = '#C8E6C9'  # Light green
                        border_color = '#388E3C'
                        style = '"filled"'
                        category = "input"
                else:
                    # Intermediate – classify by memory_category × is_gradient
                    mc = v.memory_category
                    is_grad = v.is_gradient

                    if mc is MemoryCategory.MATERIALIZED:
                        if is_grad:
                            fillcolor = '#F5F5F5'      # Very light gray
                            border_color = '#757575'    # Medium gray
                        else:
                            fillcolor = '#E0E0E0'      # Gray
                            border_color = '#9E9E9E'
                        style = '"filled"'
                        category = "saved_grad" if is_grad else "saved"

                    elif mc is MemoryCategory.NOT_MATERIALIZED:
                        if is_grad:
                            fillcolor = '#F3E5F5'      # Very light purple
                            border_color = '#7B1FA2'    # Purple
                        else:
                            fillcolor = '#EDE7F6'       # Light purple
                            border_color = '#9575CD'    # Medium purple
                        style = '"filled,dotted"'
                        category = "not_mat_grad" if is_grad else "not_mat"

                    else:  # RECOMPUTED
                        if is_grad:
                            fillcolor = '#FFF8E1'       # Very light amber
                            border_color = '#E65100'    # Deep orange
                        else:
                            fillcolor = '#FFF3E0'       # Light orange
                            border_color = '#E65100'
                        style = '"filled,dashed"'
                        category = "recomp_grad" if is_grad else "recomp"

                shape = "ellipse"
                lines.append(
                    f'    {val_id} [label="{label}", shape={shape}, '
                    f'style={style}, fillcolor="{fillcolor}", color="{border_color}", penwidth=2];'
                )
            else:
                # Token
                fillcolor = '#FFE0B2'  # Light orange
                border_color = '#F57C00'  # Orange border
                lines.append(
                    f'    {val_id} [label="{label}", shape=ellipse, '
                    f'style="filled", fillcolor="{fillcolor}", color="{border_color}", penwidth=2];'
                )
    
    # Add edges
    lines.append("")
    lines.append("    // Edges")
    
    for node in nodes:
        nid = node_ids[node.id]
        
        for v in node.inputs:
            vid = id(v)
            
            if show_tensors:
                # Edge from value to op
                val_id = value_ids[vid]
                lines.append(f'    {val_id} -> {nid};')
            else:
                # Edge from producer op to this op (skip values)
                if isinstance(v, (Tensor, Token)) and v.producer is not None:
                    if v.producer.id in node_ids:
                        src_id = node_ids[v.producer.id]
                        label = ""
                        if isinstance(v, Tensor) and v.name:
                            label = v.name
                        if label:
                            lines.append(f'    {src_id} -> {nid} [label="{label}"];')
                        else:
                            lines.append(f'    {src_id} -> {nid};')
    
    # Connect producers to values (if showing tensors)
    if show_tensors:
        for vid, v in value_objs.items():
            val_id = value_ids[vid]
            if isinstance(v, (Tensor, Token)) and v.producer is not None:
                if v.producer.id in node_ids:
                    src_id = node_ids[v.producer.id]
                    lines.append(f'    {src_id} -> {val_id};')
        
        # Add legend
        lines.append("")
        lines.append("    // Legend")
        lines.append("    subgraph cluster_legend {")
        lines.append('        label="Legend";')
        lines.append('        style="rounded";')
        lines.append('        color="#BDBDBD";')
        lines.append('        fontsize=10;')
        lines.append('        node [fontsize=9];')
        lines.append('        ')
        lines.append('        legend_param [label="Parameter", shape=ellipse, style="filled", fillcolor="#BBDEFB", color="#1976D2", penwidth=2];')
        lines.append('        legend_input [label="Input", shape=ellipse, style="filled", fillcolor="#C8E6C9", color="#388E3C", penwidth=2];')
        lines.append('        legend_saved [label="Materialized", shape=ellipse, style="filled", fillcolor="#E0E0E0", color="#9E9E9E", penwidth=2];')
        lines.append('        legend_saved_g [label="Materialized (grad)", shape=ellipse, style="filled", fillcolor="#F5F5F5", color="#757575", penwidth=2];')
        lines.append('        legend_notmat [label="Not materialized", shape=ellipse, style="filled,dotted", fillcolor="#EDE7F6", color="#9575CD", penwidth=2];')
        lines.append('        legend_notmat_g [label="Not materialized (grad)", shape=ellipse, style="filled,dotted", fillcolor="#F3E5F5", color="#7B1FA2", penwidth=2];')
        lines.append('        legend_token [label="Token", shape=ellipse, style="filled", fillcolor="#FFE0B2", color="#F57C00", penwidth=2];')
        lines.append('        ')
        lines.append('        legend_param -> legend_input -> legend_saved -> legend_saved_g -> legend_notmat -> legend_notmat_g -> legend_token [style=invis];')
        lines.append("    }")
    
    lines.append("}")
    return "\n".join(lines)


def visualize_graph(
    *roots: Union[Value, ExtractedGraph],
    output: Optional[str] = None,
    format: str = "svg",
    title: Optional[str] = None,
    show_tensors: bool = False,
    show_costs: bool = True,
    rankdir: str = "TB",
    view: bool = False,
) -> Optional[str]:
    """
    Visualize the computation graph.
    
    Args:
        roots: Tensors, Tokens, or an ExtractedGraph to visualize
        output: Output file path (without extension). If None, returns DOT string.
        format: Output format - "svg", "png", "pdf", "dot"
        title: Optional title for the graph
        show_tensors: If True, show tensor nodes as separate ellipses
        show_costs: If True, show cost metadata on op nodes
        rankdir: Graph direction - "TB" (top to bottom) or "LR" (left to right)
        view: If True, open the rendered file
    
    Returns:
        DOT string if output is None, otherwise None (writes to file)
    """
    dot_str = to_dot(
        *roots,
        title=title,
        show_tensors=show_tensors,
        show_costs=show_costs,
        rankdir=rankdir,
    )
    
    if output is None:
        return dot_str
    
    if format == "dot":
        # Just write the DOT file
        with open(output if output.endswith(".dot") else f"{output}.dot", "w") as f:
            f.write(dot_str)
        return None
    
    # Use graphviz to render
    try:
        import graphviz
    except ImportError:
        raise ImportError(
            "graphviz package required for rendering. "
            "Install with: pip install graphviz\n"
            "Also ensure graphviz is installed on your system: "
            "apt-get install graphviz (Ubuntu) or brew install graphviz (macOS)"
        )
    
    # Remove extension if present
    if output.endswith(f".{format}"):
        output = output[:-len(format)-1]
    
    source = graphviz.Source(dot_str)
    source.render(output, format=format, cleanup=True, view=view)
    
    return None


def print_graph_summary(
    *roots: Union[Value, ExtractedGraph],
    show_costs: bool = True,
) -> None:
    """
    Print a text summary of the computation graph.
    
    Args:
        roots: Tensors, Tokens, or an ExtractedGraph to summarize
        show_costs: If True, show cost metadata
    """
    # Handle ExtractedGraph input
    if len(roots) == 1 and isinstance(roots[0], ExtractedGraph):
        graph = roots[0]
    else:
        values = [r for r in roots if isinstance(r, (Tensor, Token))]
        graph = get_graph(*values)
    
    nodes = graph.nodes
    
    print(f"Computation Graph: {len(nodes)} nodes")
    print("=" * 60)
    
    # Count ops by kind
    kind_counts: Dict[str, int] = {}
    total_flops = 0
    total_mem_read = 0
    total_mem_write = 0
    
    for node in nodes:
        kind_counts[node.kind] = kind_counts.get(node.kind, 0) + 1
        if show_costs:
            try:
                cost = node.get_cost_meta()
                total_flops += cost["flops"]
                total_mem_read += cost["mem_read"]
                total_mem_write += cost["mem_write"]
            except Exception:
                pass
    
    print("\nOp counts:")
    for kind, count in sorted(kind_counts.items(), key=lambda x: -x[1]):
        print(f"  {kind}: {count}")
    
    if show_costs:
        print(f"\nTotal costs:")
        print(f"  FLOPs:     {_format_num(total_flops)}")
        print(f"  Mem read:  {_format_num(total_mem_read)} elements")
        print(f"  Mem write: {_format_num(total_mem_write)} elements")
        
        # Estimate arithmetic intensity
        total_mem = total_mem_read + total_mem_write
        if total_mem > 0:
            ai = total_flops / total_mem
            print(f"  Arithmetic Intensity: {ai:.2f} FLOPs/element")

    # Activation memory breakdown
    mem_info = get_activation_summary(graph)
    _print_activation_memory(mem_info)


def _print_activation_memory(info: ActivationMemoryInfo) -> None:
    """Print activation memory breakdown (trace-based peak analysis)."""
    if info.peak_bytes == 0 and not info.live_at_peak:
        return

    print(f"\nActivation memory (release-after-last-use):")
    print(f"  Peak memory:        {_format_bytes(info.peak_bytes)}")
    print(f"  Live tensors@peak:  {len(info.live_at_peak)}")
    if info.peak_node is not None:
        kind = info.peak_node.kind
        nid = info.peak_node.id
        print(f"  Peak at step:       {info.peak_step}  (node {nid}: {kind})")
    print(f"  Parameters:         {len(info.parameter_tensors)} tensors  (permanent)")
    print(f"  Inputs:             {len(info.input_tensors)} tensors  (permanent)")


def _format_bytes(b: int) -> str:
    """Human-readable byte count."""
    if b >= 1 << 30:
        return f"{b / (1 << 30):.2f} GiB"
    if b >= 1 << 20:
        return f"{b / (1 << 20):.2f} MiB"
    if b >= 1 << 10:
        return f"{b / (1 << 10):.2f} KiB"
    return f"{b} B"
