from pathlib import Path
import pandas as pd


class NCCLRingVisualizer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

        required_cols = {
            "commId",
            "channelId",
            "myRank",
            "prevRank",
            "nextRank",
            "nodeId",
        }
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def generate_dot(
        self,
        commId,
        channelId,
        show_pid: bool = True,
    ) -> str:
        """
        Generate Graphviz DOT string for one NCCL ring.
        """
        df = self.df[
            (self.df.commId == commId) & (self.df.channelId == channelId)
        ]

        if df.empty:
            raise ValueError(f"No data for commId={commId}, channelId={channelId}")

        lines = []
        lines.append("digraph NCCLRing {")
        lines.append("  rankdir=LR;")
        lines.append("  node [shape=circle, style=filled, fillcolor=lightgray];")
        lines.append("  edge [arrowhead=vee];")
        lines.append("")

        # ---- cluster by nodeId ----
        for nodeId, g in df.groupby("nodeId"):
            lines.append(f'  subgraph cluster_node_{nodeId} {{')
            lines.append(f'    label="node {nodeId}";')
            lines.append("    style=rounded;")
            lines.append("    color=gray;")

            for _, row in g.iterrows():
                rank = row["myRank"]
                label = f"rank {rank}"

                if show_pid and "pid" in row:
                    label += f"\\npid {row['pid']}"

                lines.append(f'    {rank} [label="{label}"];')

            lines.append("  }")
            lines.append("")

        # ---- ring edges: myRank -> nextRank ----
        for _, row in df.iterrows():
            src = row["myRank"]
            dst = row["nextRank"]
            if dst != -1:
                lines.append(f"  {src} -> {dst};")

        lines.append("}")
        return "\n".join(lines)

    def render(
        self,
        commId,
        channelId,
        out_dir="nccl_ring_topo",
        fmt="png",
        **kwargs,
    ):
        """
        Generate dot + rendered image (requires graphviz installed).
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        dot = self.generate_dot(commId, channelId, **kwargs)

        dot_path = out_dir / f"comm_{commId}_ch_{channelId}.dot"
        img_path = out_dir / f"comm_{commId}_ch_{channelId}.{fmt}"

        dot_path.write_text(dot)

        import subprocess
        subprocess.run(
            ["dot", f"-T{fmt}", str(dot_path), "-o", str(img_path)],
            check=True,
        )

        return dot_path, img_path


class NCCLTreeVisualizer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

        required_cols = {
            "commId",
            "channelId",
            "myRank",
            "parentRank",
            "child1Rank",
            "child2Rank",
            "child3Rank",
            "nodeId",
        }
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _escape(self, x):
        return str(x).replace("-", "_")

    def generate_dot(
        self,
        commId,
        channelId,
        show_pid: bool = True,
        show_parent_edge: bool = False,
    ) -> str:
        """
        Generate Graphviz DOT string for one NCCL tree.
        """
        df = self.df[
            (self.df.commId == commId) & (self.df.channelId == channelId)
        ]

        if df.empty:
            raise ValueError(f"No data for commId={commId}, channelId={channelId}")

        lines = []
        lines.append("digraph NCCLTree {")
        lines.append("  rankdir=TB;")
        lines.append("  node [shape=circle, style=filled, fillcolor=lightgray];")
        lines.append("  edge [arrowhead=vee];")
        lines.append("")

        # ---- cluster by nodeId ----
        for nodeId, g in df.groupby("nodeId"):
            lines.append(f'  subgraph cluster_node_{nodeId} {{')
            lines.append(f'    label="node {nodeId}";')
            lines.append("    style=rounded;")
            lines.append("    color=gray;")

            for _, row in g.iterrows():
                rank = row["myRank"]
                label = f"rank {rank}"
                if show_pid and "pid" in row:
                    label += f"\\npid {row['pid']}"

                lines.append(
                    f'    {rank} [label="{label}"];'
                )

            lines.append("  }")
            lines.append("")

        # ---- edges (parent -> child) ----
        for _, row in df.iterrows():
            src = row["myRank"]

            for c in ["child1Rank", "child2Rank", "child3Rank"]:
                dst = row[c]
                if dst != -1:
                    lines.append(f"  {src} -> {dst};")

            if show_parent_edge and row["parentRank"] != -1:
                parent = row["parentRank"]
                lines.append(
                    f"  {parent} -> {src} [style=dashed, color=blue];"
                )

        lines.append("}")
        return "\n".join(lines)

    def render(
        self,
        commId,
        channelId,
        out_dir="nccl_topo",
        fmt="png",
        **kwargs,
    ):
        """
        Generate dot + rendered image (requires graphviz installed).
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        dot = self.generate_dot(commId, channelId, **kwargs)

        dot_path = out_dir / f"comm_{commId}_ch_{channelId}.dot"
        img_path = out_dir / f"comm_{commId}_ch_{channelId}.{fmt}"

        dot_path.write_text(dot)

        import subprocess

        subprocess.run(
            ["dot", f"-T{fmt}", str(dot_path), "-o", str(img_path)],
            check=True,
        )

        return dot_path, img_path
