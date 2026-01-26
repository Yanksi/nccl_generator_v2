from configure_ring_topo import *
from configure_tree_topo import *
import pandas as pd
import logging
from typing import Dict, Tuple
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def get_node_groups(topo_file) -> pd.DataFrame:
    Tier_to_RadixDown = []
    cur_tier = None

    with open(topo_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("Tier"):
                _, tier_id = line.split()
                cur_tier = int(tier_id)

            elif line.startswith("Radix_Down") and cur_tier is not None:
                _, radix_down = line.split()
                Tier_to_RadixDown.append({
                    "Tier": cur_tier,
                    "Radix_Down": int(radix_down)
                })

    Tier_to_RadixDown_df = pd.DataFrame(Tier_to_RadixDown).sort_values("Tier").reset_index(drop=True)
    return Tier_to_RadixDown_df

def build_comm_tier0_groups(comm_info: pd.DataFrame, tier_0_groups: pd.DataFrame) -> pd.DataFrame:
    comm_nodes_df = comm_info[["commId", "nodeId"]].drop_duplicates().reset_index(drop=True)
    temp_df = comm_nodes_df.merge(tier_0_groups[["nodeId", "tier0_switch_id", "nodeId_in_switch_group", "num_nodes_in_switch_group"]], on="nodeId", how="left", validate="many_to_one")
    temp_df["comm_switch_groupId"] = temp_df.groupby("commId")["tier0_switch_id"].transform(lambda x: pd.factorize(x)[0])
    temp_df["num_nodes_in_comm_switch_group"] = temp_df.groupby(["commId", "comm_switch_groupId"])["nodeId"].transform("count")
    temp_df = temp_df.sort_values(by=["commId", "comm_switch_groupId", "nodeId"], ignore_index=True)
    temp_df["nodeId_in_comm_switch_group"] =  temp_df.groupby(["commId", "comm_switch_groupId"]).cumcount()

    res_df = temp_df[
        [
            "commId",
            "nodeId",
            "comm_switch_groupId",
            "nodeId_in_comm_switch_group",
            "num_nodes_in_comm_switch_group",
            "tier0_switch_id",
            "nodeId_in_switch_group",
            "num_nodes_in_switch_group"
        ]
    ]

    return res_df

def update_topo_info(comm_info: pd.DataFrame, comm_ring_info: pd.DataFrame, comm_tree_info: pd.DataFrame, node_groups: pd.DataFrame) -> Tuple[pd.DataFrame]:
    node_ids = comm_info['nodeId'].drop_duplicates().reset_index(drop=True)
    tier_0_Radix_Down = node_groups[node_groups["Tier"] == 0]["Radix_Down"].iloc[0]  ## get the radix down value of tier0 switches
    tier_0_groups =  node_ids.to_frame(name="nodeId").assign(tier0_switch_id=lambda df: df.index // tier_0_Radix_Down)  ## assign a tier0 switch id to each node
    tier_0_groups["nodeId_in_switch_group"] = tier_0_groups.groupby("tier0_switch_id").cumcount()
    tier_0_groups["num_nodes_in_switch_group"] = tier_0_groups.groupby("tier0_switch_id")["nodeId"].transform("nunique")

    comm_nodes_groups = build_comm_tier0_groups(comm_info, tier_0_groups)

    commId_nodeId_df = comm_info[["commId", "nodeId"]].drop_duplicates().reset_index(drop=True)
    commId_node_count = comm_info.groupby("commId")["nodeId"].nunique().rename("num_nodes")
    commId_gpu_count = comm_info.drop_duplicates(subset=["commId", "nodeId", "pid"]).groupby("commId").size().rename("num_gpus")  ## maybe ‘drop_duplicates(subset=["commId", "nodeId", "pid"])’ not needed
    commId_rank_to_node = comm_info[["commId", "rank", "nodeId"]].drop_duplicates().set_index(["commId", "rank"])["nodeId"]

    ## Configure Ring Topology
    ring_ingress_egress_df = get_ring_ingress_egress_df(comm_ring_info, commId_node_count, commId_rank_to_node)
    assert_ring_prev_next_consistency(ring_ingress_egress_df)

    ring_ingress_egress_df = update_ring_ingress_egress_df(ring_ingress_egress_df, comm_nodes_groups)
    assert_ring_prev_next_consistency(ring_ingress_egress_df)

    comm_ring_info = update_comm_ring_info(comm_ring_info, ring_ingress_egress_df)

    ## Configure Tree Topology
    tree_ingress_egress_df = get_tree_ingress_egress_df(comm_tree_info, commId_node_count, commId_rank_to_node)
    # tree_ingress_egress_df = get_tree_ingress_egress_df_deprecated(comm_tree_info, commId_node_count, commId_rank_to_node)
    # assert_tree_child_parent_consistency(tree_ingress_egress_df)

    tree_ingress_egress_df = update_tree_ingress_egress_df(tree_ingress_egress_df, comm_nodes_groups)

    comm_tree_info = update_comm_tree_info(comm_tree_info, tree_ingress_egress_df)

    return comm_info, comm_ring_info, comm_tree_info