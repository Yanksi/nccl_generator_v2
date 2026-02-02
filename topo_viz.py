from pathlib import Path
import pandas as pd
import subprocess


class NCCLRingVisualizer:
    def __init__(self, comm_info: pd.DataFrame, comm_ring_info: pd.DataFrame, out_dir="nccl_ring_topo", tag=None, fmt="pdf", ignore_invalid_comm=True):
        self.comm_info = comm_info.copy()
        self.comm_ring_info = comm_ring_info.copy()

        if ignore_invalid_comm:
            valid_comm_ids = self.comm_info.loc[self.comm_info["nRanks"] > 1, "commId"].unique()
            self.comm_ring_info = self.comm_ring_info[self.comm_ring_info["commId"].isin(valid_comm_ids)].reset_index(drop=True)

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.tag = "ring_" + tag if tag else "ring"
        self.fmt = fmt

        required_cols = {
            "commId",
            "channelId",
            "myRank",
            "prevRank",
            "nextRank",
            "nodeId",
        }
        missing = required_cols - set(comm_ring_info.columns)
        if missing:
            raise ValueError(f"Missing required comm_ring_info columns: {missing}")

    def generate_dot(
        self,
        commId,
        channelId,
        show_pid: bool = True,
    ) -> str:
        """
        Generate Graphviz DOT string for one NCCL ring.
        """
        comm_ring_info_df = self.comm_ring_info[(self.comm_ring_info.commId == commId) & (self.comm_ring_info.channelId == channelId)]
        if comm_ring_info_df.empty:
            raise ValueError(f"No data for commId={commId}, channelId={channelId}")

        lines = []
        lines.append("digraph NCCLRing {")
        lines.append("  rankdir=LR;")
        lines.append("  node [shape=circle, style=filled, fillcolor=lightgray];")
        lines.append("  edge [arrowhead=vee];")
        lines.append("")

        # ---- cluster by nodeId ----
        for nodeId, g in comm_ring_info_df.groupby("nodeId"):
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
        for _, row in comm_ring_info_df.iterrows():
            src = row["myRank"]
            dst = row["nextRank"]
            lines.append(f"  {src} -> {dst};")

        lines.append("}")
        return "\n".join(lines)

    def render(self, commId, channelId, **kwargs):
        """
        Generate dot + img (requirement: graphviz).
        """
        dot = self.generate_dot(commId, channelId, **kwargs)

        dot_path = self.out_dir / f"comm_{commId}_ch_{channelId}_{self.tag}.dot"
        img_path = self.out_dir / f"comm_{commId}_ch_{channelId}_{self.tag}.{self.fmt}"

        dot_path.write_text(dot)

        subprocess.run(
            ["dot", f"-T{self.fmt}", str(dot_path), "-o", str(img_path)],
            check=True,
        )

        return
    
    def render_all(self, **kwargs):
        """
        Generate dot + rendered image for all commId/channelId combinations.
        """
        for (commId, channelId), group in self.comm_ring_info.groupby(["commId", "channelId"]):
            self.render(commId=commId, channelId=channelId, show_pid=True)

        return

class NCCLTreeVisualizer:
    def __init__(self, comm_info: pd.DataFrame, comm_tree_info: pd.DataFrame, out_dir="nccl_tree_topo", tag=None, fmt="pdf", ignore_invalid_comm=True):
        self.comm_info = comm_info.copy()
        self.comm_tree_info = comm_tree_info.copy()

        if ignore_invalid_comm:
            valid_comm_ids = self.comm_info.loc[self.comm_info["nRanks"] > 1, "commId"].unique()
            self.comm_tree_info = self.comm_tree_info[self.comm_tree_info["commId"].isin(valid_comm_ids)].reset_index(drop=True)

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.tag = "tree_" + tag if tag else "tree"
        self.fmt = fmt

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
        missing = required_cols - set(comm_tree_info.columns)
        if missing:
            raise ValueError(f"Missing required comm_tree_info columns: {missing}")

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
        comm_tree_info_df = self.comm_tree_info[(self.comm_tree_info.commId == commId) & (self.comm_tree_info.channelId == channelId)]
        if comm_tree_info_df.empty:
            raise ValueError(f"No data for commId={commId}, channelId={channelId}")

        lines = []
        lines.append("digraph NCCLTree {")
        lines.append("  rankdir=TB;")
        lines.append("  node [shape=circle, style=filled, fillcolor=lightgray];")
        lines.append("  edge [arrowhead=vee];")
        lines.append("")

        # ---- cluster by nodeId ----
        for nodeId, g in comm_tree_info_df.groupby("nodeId"):
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

        # ---- tree edges (child -> parent) ----
        for _, row in comm_tree_info_df.iterrows():
            dst = row["myRank"]

            for c in ["child1Rank", "child2Rank", "child3Rank"]:
                src = row[c]
                if src != -1:
                    lines.append(f"  {src} -> {dst};")

            if show_parent_edge and row["parentRank"] != -1:
                parent = row["parentRank"]
                lines.append(
                    f"  {parent} -> {dst} [style=dashed, color=blue];"
                )

        lines.append("}")
        return "\n".join(lines)

    def render(self, commId, channelId, **kwargs):
        """
        Generate dot + img (requirement: graphviz).
        """
        dot = self.generate_dot(commId, channelId, **kwargs)

        dot_path = self.out_dir / f"comm_{commId}_ch_{channelId}_{self.tag}.dot"
        img_path = self.out_dir / f"comm_{commId}_ch_{channelId}_{self.tag}.{self.fmt}"

        dot_path.write_text(dot)

        subprocess.run(
            ["dot", f"-T{self.fmt}", str(dot_path), "-o", str(img_path)],
            check=True,
        )

        return
    
    def render_all(self, **kwargs):
        """
        Generate dot + rendered image for all commId/channelId combinations.
        """
        for (commId, channelId), group in self.comm_tree_info.groupby(["commId", "channelId"]):
            self.render(commId=commId, channelId=channelId, show_pid=True)

        return
