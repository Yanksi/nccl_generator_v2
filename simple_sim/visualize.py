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
    if hasattr(node, 'a_shape') and hasattr(node, 'b_shape'):
        m, k = node.a_shape
        _, n = node.b_shape
        lines.append(f"[{m}×{k}]×[{k}×{n}]")
    elif hasattr(node, 'q_shape') and hasattr(node, 'kv_shape'):
        B, Sq, H, D = node.q_shape
        lines.append(f"B={B}, S={Sq}")
    elif hasattr(node, 'shape') and hasattr(node, 'num_inputs'):
        from .utils import numel
        lines.append(f"n={numel(node.shape)}, in={node.num_inputs}")
    elif hasattr(node, 'bytes'):
        lines.append(f"bytes={_format_num(node.bytes)}")

    # Collectives: show parallelism group
    if hasattr(node, 'group'):
        g = node.group
        lines.append(f"<i>{g.kind.upper()} (size={g.size})</i>")

    # PP send/recv: show label, src/dst
    if hasattr(node, 'label') and node.label:
        lines.append(f"<i>{node.label}</i>")
    if hasattr(node, 'src'):
        lines.append(f"src={node.src}")
    elif hasattr(node, 'dst'):
        lines.append(f"dst={node.dst}")
    
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


# -------- Multi-rank graph visualization --------

# Rank-specific colors for subgraph backgrounds
RANK_COLORS = [
    "#E3F2FD",  # Light blue
    "#E8F5E9",  # Light green
    "#FFF3E0",  # Light orange
    "#F3E5F5",  # Light purple
    "#E0F7FA",  # Light cyan
    "#FBE9E7",  # Light deep orange
    "#E8EAF6",  # Light indigo
    "#F1F8E9",  # Light light green
]


def multi_rank_to_dot(
    multi_graph: "MultiRankGraph",
    title: Optional[str] = None,
    show_costs: bool = True,
    show_tensors: bool = False,
    rankdir: str = "TB",
    merged_mode: bool = False,
) -> str:
    """
    Generate DOT format string for a multi-rank aggregated graph.
    
    Each rank is shown as a subgraph (cluster), with cross-rank edges
    for send→recv pairs and collective groupings.
    
    Args:
        multi_graph: MultiRankGraph from aggregate_graphs()
        title: Optional title for the graph
        show_costs: If True, show cost metadata on nodes
        show_tensors: If True, show tensor nodes; if False, only show ops
        rankdir: Graph direction - "TB" (top to bottom) or "LR" (left to right)
        merged_mode: If True, visualize merged cross-rank ops with proper data flow
    
    Returns:
        DOT format string
    """
    from .aggregate import MultiRankGraph, MergedCollectiveOp
    
    lines = ["digraph G {"]
    lines.append(f'    rankdir={rankdir};')
    lines.append('    compound=true;')  # Allow edges between clusters
    lines.append('    newrank=true;')   # Better ranking with cross-cluster edges
    lines.append('    node [fontname="Helvetica", fontsize=10];')
    lines.append('    edge [fontname="Helvetica", fontsize=9];')
    
    if title:
        lines.append(f'    labelloc="t";')
        lines.append(f'    label="{title}";')
        lines.append('    fontsize=14;')
    
    # Build node ID mapping and collect all values
    node_ids = {}
    value_ids = {}  # id(v) -> "val_X"
    value_objs = {}  # id(v) -> v
    
    for rank, g in multi_graph.graphs.items():
        for node in g.nodes:
            node_ids[node.id] = f"op_{node.id}"
            # Collect input values
            for v in node.inputs:
                vid = id(v)
                if vid not in value_ids:
                    value_ids[vid] = f"val_{len(value_ids)}"
                    value_objs[vid] = v
            # Collect output values
            for v in node.outputs:
                vid = id(v)
                if vid not in value_ids:
                    value_ids[vid] = f"val_{len(value_ids)}"
                    value_objs[vid] = v
    
    # Create subgraph for each rank
    for rank in multi_graph.ranks:
        g = multi_graph.graphs[rank]
        color = RANK_COLORS[rank % len(RANK_COLORS)]
        
        lines.append("")
        lines.append(f"    subgraph cluster_rank{rank} {{")
        lines.append(f'        label="GPU {rank}";')
        lines.append(f'        style="filled,rounded";')
        lines.append(f'        fillcolor="{color}";')
        lines.append(f'        color="#BDBDBD";')
        lines.append('        fontsize=12;')
        lines.append("")
        
        # Add nodes for this rank (filter by actual rank, skip MergedCollectiveOp)
        for node in g.nodes:
            # Skip MergedCollectiveOp - rendered separately outside clusters
            if isinstance(node, MergedCollectiveOp):
                continue
            # Only include nodes that belong to this rank
            # (skip nodes with rank=None or wrong rank)
            if node.rank != rank:
                continue
                
            nid = node_ids[node.id]
            node_color = _get_node_color(node)
            label = _get_node_label(node) if show_costs else f"<b>{_get_display_kind(node)}</b><br/>id={node.id}"
            
            # Add rank info to label
            label = f"<b>{_get_display_kind(node)}</b><br/>id={node.id}"
            if show_costs:
                try:
                    cost = node.get_cost_meta()
                    if cost["flops"] > 0 or cost["mem_read"] > 0:
                        label += f"<br/>flops={_format_num(cost['flops'])}"
                except Exception:
                    pass
                if hasattr(node, 'bytes'):
                    label += f"<br/>bytes={_format_num(node.bytes)}"
            
            # Comm info
            if hasattr(node, 'group'):
                label += f"<br/><i>{node.group.match_key}</i>"
            if hasattr(node, 'dst'):
                label += f"<br/>dst={node.dst}"
            if hasattr(node, 'src') and node.src >= 0:
                label += f"<br/>src={node.src}"
            
            # Determine shape
            if isinstance(node, CommOp):
                shape = "hexagon"
            else:
                shape = "box"
            
            lines.append(
                f'        {nid} [label=<{label}>, shape={shape}, '
                f'style="filled,rounded", fillcolor="{node_color}", fontcolor="white"];'
            )
        
        # Add tensor nodes for this rank if requested
        if show_tensors:
            lines.append("")
            lines.append("        // Value nodes (tensors/tokens)")
            for vid, v in value_objs.items():
                # Only include tensors that belong to this rank
                tensor_rank = None
                if hasattr(v, 'producer') and v.producer is not None:
                    if isinstance(v.producer, MergedCollectiveOp):
                        # Skip merged collective outputs - they'll be connected directly
                        # from the merged op to consumers to avoid cross-cluster issues
                        continue
                    else:
                        tensor_rank = v.producer.rank
                else:
                    # No producer (input/parameter) - find a consumer in this rank
                    for node in g.nodes:
                        if node.rank == rank and v in node.inputs:
                            tensor_rank = rank
                            break
                
                if tensor_rank != rank:
                    continue
                    
                val_id = value_ids[vid]
                label = _get_value_label(v)
                
                if isinstance(v, Tensor):
                    has_producer_in_graph = (
                        v.producer is not None and v.producer.id in node_ids
                    )
                    
                    if not has_producer_in_graph:
                        if v.requires_grad:
                            fillcolor = '#BBDEFB'  # Light blue (parameter)
                            border_color = '#1976D2'
                        else:
                            fillcolor = '#C8E6C9'  # Light green (input)
                            border_color = '#388E3C'
                    else:
                        mc = v.memory_category
                        is_grad = v.is_gradient
                        if mc is MemoryCategory.MATERIALIZED:
                            fillcolor = '#F5F5F5' if is_grad else '#E0E0E0'
                            border_color = '#757575' if is_grad else '#9E9E9E'
                        elif mc is MemoryCategory.NOT_MATERIALIZED:
                            fillcolor = '#F3E5F5' if is_grad else '#EDE7F6'
                            border_color = '#7B1FA2' if is_grad else '#9575CD'
                        else:  # RECOMPUTED
                            fillcolor = '#FFF8E1' if is_grad else '#FFF3E0'
                            border_color = '#E65100'
                    
                    lines.append(
                        f'        {val_id} [label="{label}", shape=ellipse, '
                        f'style="filled", fillcolor="{fillcolor}", color="{border_color}", penwidth=2];'
                    )
                else:
                    # Token
                    lines.append(
                        f'        {val_id} [label="{label}", shape=ellipse, '
                        f'style="filled", fillcolor="#FFE0B2", color="#F57C00", penwidth=2];'
                    )
        
        # Add intra-rank edges
        lines.append("")
        lines.append("        // Edges")
        for node in g.nodes:
            # Skip MergedCollectiveOp - handled separately
            if isinstance(node, MergedCollectiveOp):
                continue
            # Skip nodes not belonging to this rank
            if node.rank != rank:
                continue
                
            nid = node_ids[node.id]
            for inp in node.inputs:
                if show_tensors:
                    # Skip merged collective outputs - edges come from merged op directly
                    if hasattr(inp, 'producer') and isinstance(inp.producer, MergedCollectiveOp):
                        continue
                    vid = id(inp)
                    if vid in value_ids:
                        # Determine which rank this tensor belongs to
                        tensor_rank = None
                        if hasattr(inp, 'producer') and inp.producer is not None:
                            tensor_rank = inp.producer.rank
                        else:
                            # No producer (input/parameter) - belongs to the rank that consumes it
                            tensor_rank = rank
                        
                        if tensor_rank == rank:
                            val_id = value_ids[vid]
                            lines.append(f'        {val_id} -> {nid};')
                else:
                    if hasattr(inp, 'producer') and inp.producer is not None:
                        # Skip MergedCollectiveOp - edges drawn separately
                        if isinstance(inp.producer, MergedCollectiveOp):
                            continue
                        if inp.producer.id in node_ids:
                            src_id = node_ids[inp.producer.id]
                            if merged_mode:
                                # In merged mode, include cross-rank edges inline
                                lines.append(f'        {src_id} -> {nid};')
                            else:
                                # Only add edge if producer is in same rank
                                if inp.producer.rank == rank:
                                    lines.append(f'        {src_id} -> {nid};')
        
        # Connect producers to values (if showing tensors)
        if show_tensors:
            for vid, v in value_objs.items():
                val_id = value_ids[vid]
                if isinstance(v, (Tensor, Token)) and v.producer is not None:
                    # Skip MergedCollectiveOp - edges handled separately
                    if isinstance(v.producer, MergedCollectiveOp):
                        continue
                    if v.producer.id in node_ids:
                        if v.producer.rank == rank:
                            src_id = node_ids[v.producer.id]
                            lines.append(f'        {src_id} -> {val_id};')
        
        lines.append("    }")
    
    # Add merged collective nodes outside clusters (they span multiple ranks)
    merged_collective_ids = set()
    for coll in multi_graph.cross_collectives:
        if hasattr(coll, 'merged_op') and coll.merged_op is not None and merged_mode:
            merged_op = coll.merged_op
            # Use consistent ID with node_ids
            mid = node_ids[merged_op.id]
            merged_collective_ids.add(mid)
            
            # Draw merged collective as a special node spanning ranks
            ranks_str = ", ".join(str(r) for r in merged_op.participating_ranks)
            label = f"<b>Merged {merged_op.kind}</b><br/>ranks: [{ranks_str}]"
            if show_costs:
                label += f"<br/>bytes={_format_num(merged_op.bytes)}"
            
            lines.append("")
            lines.append(f'    {mid} [label=<{label}>, shape=octagon, ')
            lines.append(f'        style="filled,bold", fillcolor="#7B1FA2", fontcolor="white", penwidth=3];')
            
            # Add edges from inputs to merged node
            for inp in merged_op.inputs:
                if show_tensors:
                    # Connect through tensor node
                    vid = id(inp)
                    if vid in value_ids:
                        lines.append(f'    {value_ids[vid]} -> {mid} [color="#9C27B0", penwidth=2, constraint=true];')
                else:
                    # Connect from producer op directly
                    if hasattr(inp, 'producer') and inp.producer is not None:
                        if inp.producer.id in node_ids:
                            src_id = node_ids[inp.producer.id]
                            lines.append(f'    {src_id} -> {mid} [color="#9C27B0", penwidth=2, constraint=true];')
            
            # Add edges from merged node to outputs' consumers
            # Always connect directly to consumers (skip tensor nodes to avoid cross-cluster edge issues)
            for out in merged_op.outputs:
                # Find consumers of this output tensor and draw edges directly
                for rank, g in multi_graph.graphs.items():
                    for node in g.nodes:
                        if isinstance(node, MergedCollectiveOp):
                            continue
                        if out in node.inputs:
                            consumer_id = node_ids[node.id]
                            lines.append(f'    {mid} -> {consumer_id} [color="#9C27B0", penwidth=2, constraint=true];')
    
    # Add cross-rank edges for send→recv pairs
    if merged_mode:
        # In merged mode, draw send→recv as data-flow edges (solid line)
        lines.append("")
        lines.append("    // Cross-rank send->recv edges (merged)")
        for send_op, recv_op, src_rank, dst_rank in multi_graph.send_recv_pairs:
            send_id = node_ids[send_op.id]
            recv_id = node_ids[recv_op.id]
            lines.append(
                f'    {send_id} -> {recv_id} '
                f'[color="#D32F2F", penwidth=2, '
                f'label="data", fontcolor="#D32F2F"];'
            )
    else:
        lines.append("")
        lines.append("    // Cross-rank send->recv edges")
        for send_op, recv_op, src_rank, dst_rank in multi_graph.send_recv_pairs:
            send_id = node_ids[send_op.id]
            recv_id = node_ids[recv_op.id]
            lines.append(
                f'    {send_id} -> {recv_id} '
                f'[style="dashed", color="#D32F2F", penwidth=2, '
                f'label="P2P", fontcolor="#D32F2F", constraint=true];'
            )
        
        # Add visual grouping for collectives (connect them with invisible edges + label)
        lines.append("")
        lines.append("    // Collective groupings")
        for coll in multi_graph.cross_collectives:
            if len(coll.participants) > 1:
                # Draw edges between participants to show they're synchronized
                participants = sorted(coll.participants, key=lambda x: x[0])
                for i in range(len(participants) - 1):
                    rank1, op1 = participants[i]
                    rank2, op2 = participants[i + 1]
                    id1 = node_ids[op1.id]
                    id2 = node_ids[op2.id]
                    # Bidirectional edge to show sync
                    lines.append(
                        f'    {id1} -> {id2} '
                        f'[style="dotted", color="#9C27B0", penwidth=2, '
                        f'dir="both", label="{coll.kind}", fontcolor="#9C27B0", constraint=true];'
                    )
    
    lines.append("}")
    return "\n".join(lines)


def visualize_multi_rank_graph(
    multi_graph: "MultiRankGraph",
    output: Optional[str] = None,
    format: str = "svg",
    title: Optional[str] = None,
    show_costs: bool = True,
    show_tensors: bool = False,
    merged_mode: bool = False,
    rankdir: str = "TB",
    view: bool = False,
) -> Optional[str]:
    """
    Visualize an aggregated multi-rank computation graph.
    
    Each GPU rank is shown as a separate subgraph with cross-rank
    edges for send→recv and collective synchronizations.
    
    Args:
        multi_graph: MultiRankGraph from aggregate_graphs()
        output: Output file path (without extension). If None, returns DOT string.
        format: Output format - "svg", "png", "pdf", "dot"
        title: Optional title for the graph
        show_costs: If True, show cost metadata on op nodes
        show_tensors: If True, show tensor nodes; if False, only show ops
        merged_mode: If True, visualize with merged cross-rank ops (requires merge_cross_rank=True in aggregate_graphs)
        rankdir: Graph direction - "TB" (top to bottom) or "LR" (left to right)
        view: If True, open the rendered file
    
    Returns:
        DOT string if output is None, otherwise None (writes to file)
    """
    dot_str = multi_rank_to_dot(
        multi_graph,
        title=title,
        show_costs=show_costs,
        show_tensors=show_tensors,
        rankdir=rankdir,
        merged_mode=merged_mode,
    )
    
    if output is None:
        return dot_str
    
    if format == "dot":
        with open(output if output.endswith(".dot") else f"{output}.dot", "w") as f:
            f.write(dot_str)
        return None
    
    try:
        import graphviz
    except ImportError:
        raise ImportError(
            "graphviz package required for rendering. "
            "Install with: pip install graphviz\n"
            "Also ensure graphviz is installed on your system: "
            "apt-get install graphviz (Ubuntu) or brew install graphviz (macOS)"
        )
    
    if output.endswith(f".{format}"):
        output = output[:-len(format)-1]
    
    source = graphviz.Source(dot_str)
    source.render(output, format=format, cleanup=True, view=view)
    
    return None
