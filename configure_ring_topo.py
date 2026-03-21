#%%
import pandas as pd
import logging
from typing import Dict, Tuple
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


#%%
def assert_ring_prev_next_consistency(df):
    """
    Assert that for each (commId, channelId),
    prev(X) = Y  <=>  next(Y) = X  at node level.
    """

    for (comm_id, channel_id), group in df.groupby(["commId", "channelId"]):
        # nodeId -> nextRank_nodeId
        node_to_next = dict(
            zip(group["nodeId"], group["nextRank_nodeId"])
        )

        for idx, row in group.iterrows():
            node = row["nodeId"]
            prev_node = row["prevRank_nodeId"]

            assert prev_node in node_to_next, (
                f"[Ring error] prev node missing\n"
                f"  commId={comm_id}\n"
                f"  channelId={channel_id}\n"
                f"  nodeId={node}\n"
                f"  prevRank_nodeId={prev_node}\n"
                f"  ring_nodes={list(node_to_next.keys())}"
            )

            expected = node_to_next[prev_node]

            assert expected == node, (
                f"[Ring error] prev-next mismatch\n"
                f"  commId={comm_id}\n"
                f"  channelId={channel_id}\n"
                f"  nodeId={node}\n"
                f"  prevRank_nodeId={prev_node}\n"
                f"  next(prev)={expected}\n"
                f"  expected={node}"
            )

def get_ring_ingress_egress_df(comm_ring_info: pd.DataFrame, commId_node_count: pd.DataFrame, commId_rank_to_node: pd.DataFrame) -> pd.DataFrame:
    ingress_list = []
    egress_list = []

    for line_idx in comm_ring_info.index:
        row = comm_ring_info.loc[line_idx]
        if commId_node_count[row['commId']] <= 1:
            continue
        comm_id = row['commId']
        channel_id = row['channelId']
        myRank = row['myRank']
        prevRank = row['prevRank']
        nextRank = row['nextRank']

        myRank_nodeId = commId_rank_to_node.loc[(comm_id, myRank)]
        prevRank_nodeId = commId_rank_to_node.loc[(comm_id, prevRank)]
        nextRank_nodeId = commId_rank_to_node.loc[(comm_id, nextRank)]
        if myRank_nodeId != prevRank_nodeId:
            ingress_list.append({
                "commId": comm_id,
                "nodeId": myRank_nodeId,
                "channelId": channel_id,
                "ingressRank": myRank,
                "prevRank": prevRank,
                "prevRank_nodeId": prevRank_nodeId
            })
        if myRank_nodeId != nextRank_nodeId:
            egress_list.append({
                "commId": comm_id,
                "nodeId": myRank_nodeId,
                "channelId": channel_id,
                "egressRank": myRank,
                "nextRank": nextRank,
                "nextRank_nodeId": nextRank_nodeId
            })

    ingress_df = pd.DataFrame(ingress_list).sort_values(by=["commId", "channelId", "nodeId"], ascending=[True, True, True], ignore_index=True)
    egress_df = pd.DataFrame(egress_list).sort_values(by=["commId", "channelId", "nodeId"], ascending=[True, True, True], ignore_index=True)
    ring_ingress_egress_df = ingress_df.merge(egress_df, on=["commId", "channelId", "nodeId"], how="inner", suffixes=("_ingress", "_egress"))

    return ring_ingress_egress_df

def get_ring_same_switch_neighbor_node_id(comm_nodes_groups, comm_id, node_id, local_node_id_offset):
    row = comm_nodes_groups.loc[(comm_nodes_groups["commId"] == comm_id) & (comm_nodes_groups["nodeId"] == node_id)].iloc[0]

    comm_switch_group_id = row["comm_switch_groupId"]
    comm_switch_group_size = row["num_nodes_in_comm_switch_group"]
    expected_node_local_id = (row["nodeId_in_comm_switch_group"] + local_node_id_offset + comm_switch_group_size) % comm_switch_group_size
    
    expected_row = comm_nodes_groups.loc[(comm_nodes_groups["commId"] == comm_id) & (comm_nodes_groups["comm_switch_groupId"] == comm_switch_group_id) & (comm_nodes_groups["nodeId_in_comm_switch_group"] == expected_node_local_id)].iloc[0]

    return expected_row["nodeId"]

def get_ring_neighbor_switch_node_id(comm_nodes_groups, comm_id, channel_id, node_id, switch_id_offset, local_id_offset):
    row = comm_nodes_groups.loc[(comm_nodes_groups["commId"] == comm_id) & (comm_nodes_groups["nodeId"] == node_id)].iloc[0]

    num_switch_groups = comm_nodes_groups.loc[comm_nodes_groups["commId"] == comm_id]["comm_switch_groupId"].nunique()
    expected_comm_switch_group_id = (row["comm_switch_groupId"] + switch_id_offset + num_switch_groups) % num_switch_groups

    expected_comm_switch_group_first_node_row = comm_nodes_groups.loc[(comm_nodes_groups["commId"] == comm_id) & (comm_nodes_groups["comm_switch_groupId"] == expected_comm_switch_group_id) & (comm_nodes_groups["nodeId_in_comm_switch_group"] == 0)].iloc[0]
    expected_comm_switch_group_num_nodes = expected_comm_switch_group_first_node_row["num_nodes_in_comm_switch_group"]
    expected_node_local_id = (channel_id + local_id_offset + expected_comm_switch_group_num_nodes) % expected_comm_switch_group_num_nodes
    
    expected_row = comm_nodes_groups.loc[(comm_nodes_groups["commId"] == comm_id) & (comm_nodes_groups["comm_switch_groupId"] == expected_comm_switch_group_id) & (comm_nodes_groups["nodeId_in_comm_switch_group"] == expected_node_local_id)].iloc[0]

    # print(f"num_switch_groups: {num_switch_groups}, switch_id: {switch_id}, group_size: {group_size}, switch_id_offset: {switch_id_offset}, local_id: {local_id}, expected: {expected_row["nodeId"]}")

    return expected_row["nodeId"]

def update_ring_ingress_egress_df(ring_ingress_egress_df: pd.DataFrame, comm_nodes_groups: pd.DataFrame) -> pd.DataFrame:
    ## First update within each switch group
    for line_idx in ring_ingress_egress_df.index:
        row = ring_ingress_egress_df.loc[line_idx]
        comm_id = row['commId']
        channel_id = row['channelId']
        node_id = row['nodeId']

        new_prevRank_nodeId = get_ring_same_switch_neighbor_node_id(comm_nodes_groups, comm_id, node_id, -1)
        new_nextRank_nodeId = get_ring_same_switch_neighbor_node_id(comm_nodes_groups, comm_id, node_id, 1)
        # print(f"node_id: {node_id}, new_prevRank_nodeId: {new_prevRank_nodeId}, new_nextRank_nodeId: {new_nextRank_nodeId}")

        ring_ingress_egress_df.loc[line_idx, 'prevRank_nodeId'] = new_prevRank_nodeId
        ring_ingress_egress_df.loc[line_idx, 'nextRank_nodeId'] = new_nextRank_nodeId

        ring_ingress_egress_df.loc[line_idx, 'prevRank'] = ring_ingress_egress_df.loc[
            (ring_ingress_egress_df['commId'] == comm_id) &
            (ring_ingress_egress_df['channelId'] == channel_id) &
            (ring_ingress_egress_df['nodeId'] == new_prevRank_nodeId)
        ].iloc[0]['egressRank']

        ring_ingress_egress_df.loc[line_idx, 'nextRank'] = ring_ingress_egress_df.loc[
            (ring_ingress_egress_df['commId'] == comm_id) &
            (ring_ingress_egress_df['channelId'] == channel_id) &
            (ring_ingress_egress_df['nodeId'] == new_nextRank_nodeId)
        ].iloc[0]['ingressRank']

    ## Then update across switch groups if there are more than one switch group
    for line_idx in ring_ingress_egress_df.index:
        row = ring_ingress_egress_df.loc[line_idx]
        comm_id = row['commId']
        channel_id = row['channelId']
        node_id = row['nodeId']

        if(comm_nodes_groups.loc[comm_nodes_groups["commId"] == comm_id]["comm_switch_groupId"].nunique() > 1):
            comm_switch_group_id = comm_nodes_groups.loc[(comm_nodes_groups["commId"] == comm_id) & (comm_nodes_groups["nodeId"] == node_id)].iloc[0]["comm_switch_groupId"]
            local_id = comm_nodes_groups.loc[(comm_nodes_groups["commId"] == comm_id) & (comm_nodes_groups["nodeId"] == node_id)].iloc[0]["nodeId_in_comm_switch_group"]
            num_local_nodes = comm_nodes_groups.loc[(comm_nodes_groups["commId"] == comm_id) & (comm_nodes_groups["nodeId"] == node_id)].iloc[0]["num_nodes_in_comm_switch_group"]

            if local_id == channel_id % num_local_nodes:  ## egress to the switch group, modify next, comm_switch_group_id: my - 1, locai_id: ch + 1 
                new_nextRank_nodeId = get_ring_neighbor_switch_node_id(comm_nodes_groups, comm_id, channel_id, node_id, -1, 1)

                ring_ingress_egress_df.loc[line_idx, 'nextRank_nodeId'] = new_nextRank_nodeId
                ring_ingress_egress_df.loc[line_idx, 'nextRank'] = ring_ingress_egress_df.loc[
                    (ring_ingress_egress_df['commId'] == comm_id) &
                    (ring_ingress_egress_df['channelId'] == channel_id) &
                    (ring_ingress_egress_df['nodeId'] == new_nextRank_nodeId)
                ].iloc[0]['ingressRank']

            if local_id == (channel_id + 1) % num_local_nodes:  ## ingress to the switch group, modify prev, comm_switch_group_id: my - 1, locai_id: ch
                new_prevRank_nodeId = get_ring_neighbor_switch_node_id(comm_nodes_groups, comm_id, channel_id, node_id, 1, 0)

                ring_ingress_egress_df.loc[line_idx, 'prevRank_nodeId'] = new_prevRank_nodeId
                ring_ingress_egress_df.loc[line_idx, 'prevRank'] = ring_ingress_egress_df.loc[
                    (ring_ingress_egress_df['commId'] == comm_id) &
                    (ring_ingress_egress_df['channelId'] == channel_id) &
                    (ring_ingress_egress_df['nodeId'] == new_prevRank_nodeId)
                ].iloc[0]['egressRank']

    return ring_ingress_egress_df

def update_comm_ring_info(comm_ring_info: pd.DataFrame, ring_ingress_egress_df: pd.DataFrame) -> pd.DataFrame:
    for line_idx in ring_ingress_egress_df.index:
        row = ring_ingress_egress_df.loc[line_idx]
        comm_id = row['commId']
        channel_id = row['channelId']
        node_id = row['nodeId']
        ingressRank = row['ingressRank']
        prevRank = row['prevRank']
        egressRank = row['egressRank']
        nextRank = row['nextRank']

        mask_ingress = (
                (comm_ring_info['commId'] == comm_id) &
                (comm_ring_info['channelId'] == channel_id) &
                (comm_ring_info['nodeId'] == node_id) &
                (comm_ring_info['myRank'] == ingressRank)
            )

        comm_ring_info.loc[mask_ingress, 'prevRank'] = prevRank

        mask_egress = (
                (comm_ring_info['commId'] == comm_id) &
                (comm_ring_info['channelId'] == channel_id) &
                (comm_ring_info['nodeId'] == node_id) &
                (comm_ring_info['myRank'] == egressRank)
            )

        comm_ring_info.loc[mask_egress, 'nextRank'] = nextRank

    return comm_ring_info
