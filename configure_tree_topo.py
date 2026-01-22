#%%
import pandas as pd
import logging
from typing import Dict, Tuple
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


#%%
def assert_tree_child_parent_consistency(df):
    """
    Assert that for each (commId, channelId),
    prev(X) = Y  <=>  next(Y) = X  at node level.
    """

    for (comm_id, channel_id), group in df.groupby(["commId", "channelId"]):
        # nodeId -> nextRank_nodeId
        node_to_parent = dict(
            zip(group["nodeId"], group["parentRank_nodeId"])
        )

        for idx, row in group.iterrows():
            node = row["nodeId"]
            # child1_node = row["child1Rank_nodeId"]
            child2_node = row["child2Rank_nodeId"]
            child3_node = row["child3Rank_nodeId"]

            # for child_node in [child1_node, child2_node, child3_node]:
            for child_node in [child2_node, child3_node]:
                if child_node == "-1":
                    continue
                assert child_node in node_to_parent, (
                    f"[Tree error] child node missing\n"
                    f"  commId={comm_id}\n"
                    f"  channelId={channel_id}\n"
                    f"  nodeId={node}\n"
                    f"  childRank_nodeId={child_node}\n"
                    f"  tree_nodes={list(node_to_parent.keys())}"
                )

                expected = node_to_parent[child_node]

                assert expected == node, (
                    f"[Tree error] prev-next mismatch\n"
                    f"  commId={comm_id}\n"
                    f"  channelId={channel_id}\n"
                    f"  nodeId={node}\n"
                    f"  childRank_nodeId={child_node}\n"
                    f"  parent(child)={expected}\n"
                    f"  expected={node}"
                )

# def get_tree_ingress_egress_df_deprecated(comm_tree_info: pd.DataFrame, commId_node_count: pd.DataFrame, commId_rank_to_node: pd.DataFrame) -> pd.DataFrame:
#     ingress_list = []
#     egress_list = []

#     for line_idx in comm_tree_info.index:
#         row = comm_tree_info.loc[line_idx]
#         if commId_node_count[row['commId']] <= 1:
#             continue
#         comm_id = row['commId']
#         channel_id = row['channelId']
#         myRank = row['myRank']
#         child1Rank = row['child1Rank']
#         child2Rank = row['child2Rank']
#         child3Rank = row['child3Rank']
#         parentRank = row['parentRank']
#         # print(f"helloworld: {type(child1Rank)}, helloworld2: {1 if child1Rank == -1 else 0}")

#         myRank_nodeId = commId_rank_to_node.loc[(comm_id, myRank)]
#         child1Rank_nodeId = '-1' if child1Rank == -1 else commId_rank_to_node.loc[(comm_id, child1Rank)]
#         child2Rank_nodeId = '-1' if child2Rank == -1 else commId_rank_to_node.loc[(comm_id, child2Rank)]
#         child3Rank_nodeId = '-1' if child3Rank == -1 else commId_rank_to_node.loc[(comm_id, child3Rank)]
#         parentRank_nodeId = '-1' if parentRank == -1 else commId_rank_to_node.loc[(comm_id, parentRank)]

#         if child1Rank_nodeId != '-1' and myRank_nodeId != child1Rank_nodeId:
#             ingress_list.append({
#                 "commId": comm_id,
#                 "nodeId": myRank_nodeId,
#                 "channelId": channel_id,
#                 "ingressRank_1": myRank,
#                 "child1Rank": child1Rank,
#                 "child1Rank_nodeId": child1Rank_nodeId
#             })
#         if child2Rank_nodeId != '-1' and myRank_nodeId != child2Rank_nodeId:
#             ingress_list.append({
#                 "commId": comm_id,
#                 "nodeId": myRank_nodeId,
#                 "channelId": channel_id,
#                 "ingressRank_2": myRank,
#                 "child2Rank": child2Rank,
#                 "child2Rank_nodeId": child2Rank_nodeId
#             })
#         if child3Rank_nodeId != '-1' and myRank_nodeId != child3Rank_nodeId:
#             ingress_list.append({
#                 "commId": comm_id,
#                 "nodeId": myRank_nodeId,
#                 "channelId": channel_id,
#                 "ingressRank_3": myRank,
#                 "child3Rank": child3Rank,
#                 "child3Rank_nodeId": child3Rank_nodeId
#             })
#         if parentRank_nodeId != '-1' and myRank_nodeId != parentRank_nodeId:
#             egress_list.append({
#                 "commId": comm_id,
#                 "nodeId": myRank_nodeId,
#                 "channelId": channel_id,
#                 "egressRank": myRank,
#                 "parentRank": parentRank,
#                 "parentRank_nodeId": parentRank_nodeId
#             })

#     ingress_df = pd.DataFrame(ingress_list).sort_values(by=["commId", "channelId", "nodeId"], ascending=[True, True, True], ignore_index=True)
#     egress_df = pd.DataFrame(egress_list).sort_values(by=["commId", "channelId", "nodeId"], ascending=[True, True, True], ignore_index=True)
#     # tree_ingress_egress_df = ingress_df.merge(egress_df, on=["commId", "channelId", "nodeId"], how="outer", suffixes=("_ingress", "_egress"))
#     tree_ingress_egress_df = (
#         pd.concat([ingress_df, egress_df], ignore_index=True)
#         .groupby(["commId", "channelId", "nodeId"], as_index=False)
#         .agg(lambda x: x.dropna().iloc[0] if not x.dropna().empty else pd.NA)
#         .sort_values(by=["commId", "channelId", "nodeId"], ignore_index=True)
#         )   

#     int_cols = [
#         "ingressRank_1",
#         "ingressRank_2",
#         "ingressRank_3",
#         "egressRank",
#         "child1Rank",
#         "child2Rank",
#         "child3Rank",
#         "parentRank",
#     ]

#     string_cols = [
#         "child1Rank_nodeId",
#         "child2Rank_nodeId",
#         "child3Rank_nodeId",
#         "parentRank_nodeId"
#     ]

#     for col in int_cols:
#         if col not in tree_ingress_egress_df.columns:
#             tree_ingress_egress_df[col] = -1
#         else:
#             tree_ingress_egress_df[col] = tree_ingress_egress_df[col].fillna(-1)

#         tree_ingress_egress_df[col] = tree_ingress_egress_df[col].astype("Int64")

#     for col in string_cols:
#         if col not in tree_ingress_egress_df.columns:
#             tree_ingress_egress_df[col] = -1
#         else:
#             tree_ingress_egress_df[col] = tree_ingress_egress_df[col].fillna(-1)

#         tree_ingress_egress_df[col] = tree_ingress_egress_df[col].astype("string")

#     return tree_ingress_egress_df

def get_tree_ingress_egress_df(comm_tree_info: pd.DataFrame, commId_node_count: pd.DataFrame, commId_rank_to_node: pd.DataFrame) -> pd.DataFrame:
    ingress_list = []
    egress_list = []

    for line_idx in comm_tree_info.index:
        row = comm_tree_info.loc[line_idx]
        if commId_node_count[row['commId']] <= 1:
            continue
        comm_id = row['commId']
        channel_id = row['channelId']
        myRank = row['myRank']

        parentRank = row['parentRank']
        child1Rank = row['child1Rank']
        # print(f"helloworld: {type(child1Rank)}, helloworld2: {1 if child1Rank == -1 else 0}")

        myRank_nodeId = commId_rank_to_node.loc[(comm_id, myRank)]
        parentRank_nodeId = '-1' if parentRank == -1 else commId_rank_to_node.loc[(comm_id, parentRank)]
        child1Rank_nodeId = '-1' if child1Rank == -1 else commId_rank_to_node.loc[(comm_id, child1Rank)]

        ingressMaybeRank_0 = myRank
        if child1Rank == -1 or child1Rank_nodeId != myRank_nodeId:
            ingressMaybeRank_1 = myRank
        else:
            ingressMaybeRank_1 = child1Rank

        if parentRank_nodeId == '-1' or myRank_nodeId != parentRank_nodeId:
            egress_list.append({
                "commId": comm_id,
                "nodeId": myRank_nodeId,
                "channelId": channel_id,
                "egressRank": ingressMaybeRank_0,
                "ingressMaybeRank_0": myRank,
                "ingressMaybeRank_1": ingressMaybeRank_1,
            })

    tree_ingress_egress_df = pd.DataFrame(egress_list).sort_values(by=["commId", "channelId", "nodeId"], ascending=[True, True, True], ignore_index=True)

    return tree_ingress_egress_df

def getBtree(nranks: int, rank: int):
    # Find lowest set bit in rank
    bit = 1
    while bit < nranks:
        if bit & rank:
            break
        bit <<= 1

    # Root
    if rank == 0:
        u = -1
        d0 = -1
        d1 = (bit >> 1) if nranks > 1 else -1
        parentChildType = None
        return u, d0, d1, parentChildType

    # Parent
    up = (rank ^ bit) | (bit << 1)
    if up >= nranks:
        up = (rank ^ bit)

    parentChildType = 0 if rank < up else 1
    u = up

    # Children
    lowbit = bit >> 1

    if lowbit == 0:
        d0 = -1
        d1 = -1
    else:
        d0 = rank - lowbit
        d1 = rank + lowbit

        # Make sure d1 is within bounds
        while d1 >= nranks:
            lowbit >>= 1
            if lowbit == 0:
                d1 = -1
                break
            d1 = rank + lowbit

    return u, d0, d1, parentChildType

def getDtree(nranks: int, rank: int):
    # First tree: plain btree
    s0, d0_0, d0_1, parentChildType0 = getBtree(nranks, rank)

    # Second tree: mirror or shift
    if nranks % 2 == 1:
        # shift
        shiftrank = (rank - 1 + nranks) % nranks
        u, d0, d1, parentChildType1 = getBtree(nranks, shiftrank)

        s1 = -1 if u == -1 else (u + 1) % nranks
        d1_0 = -1 if d0 == -1 else (d0 + 1) % nranks
        d1_1 = -1 if d1 == -1 else (d1 + 1) % nranks
    else:
        # mirror
        mirrorrank = nranks - 1 - rank
        u, d0, d1, parentChildType1 = getBtree(nranks, mirrorrank)

        s1 = -1 if u == -1 else nranks - 1 - u
        d1_0 = -1 if d0 == -1 else nranks - 1 - d0
        d1_1 = -1 if d1 == -1 else nranks - 1 - d1

    return (
        s0, d0_0, d0_1, parentChildType0,
        s1, d1_0, d1_1, parentChildType1
    )  ## return shifted node id within the comm switch group

def getDtreeRoots(nranks: int):
    # primal tree root is always 0
    r0 = 0

    # second tree root:
    # - if nranks is odd and > 1: 1
    # - else: nranks - 1
    if nranks % 2 == 1 and nranks > 1:
        r1 = 1
    else:
        r1 = nranks - 1

    return r0, r1  ## r0: root of the first (primal) tree, r1: root of the second tree

def commSwitchGroupNodeId_to_shiftedNodeId(node_id, root, num_nodes_in_comm_switch_group):
    return (node_id - root + num_nodes_in_comm_switch_group) % num_nodes_in_comm_switch_group

def shiftedNodeId_to_commSwitchGroupNodeId(shifted, root, num_nodes_in_comm_switch_group):
    if shifted == -1: return -1
    return (shifted + root + num_nodes_in_comm_switch_group) % num_nodes_in_comm_switch_group

def commSwitchGroupNodeId_to_globalNodeId(node_id_in_comm_switch_group, comm_nodes_groups_2, comm_id, comm_switch_group_id):
    if node_id_in_comm_switch_group == -1: return -1
    return comm_nodes_groups_2.loc[(comm_id, comm_switch_group_id, node_id_in_comm_switch_group)]["nodeId"]


def update_tree_ingress_egress_df(tree_ingress_egress_df: pd.DataFrame, comm_nodes_groups: pd.DataFrame) -> pd.DataFrame:
    comm_nodes_groups_1 = comm_nodes_groups.set_index(["commId", "nodeId"])
    comm_nodes_groups_2 = comm_nodes_groups.set_index(["commId", "comm_switch_groupId", "nodeId_in_comm_switch_group"])

    comm_channel_tree_cnt_df = tree_ingress_egress_df.groupby("commId")["channelId"].nunique().reset_index(name="numChannels")
    comm_channel_tree_cnt_df["nTrees1"] = comm_channel_tree_cnt_df["numChannels"] // 2
    comm_channel_tree_cnt_df["nTrees0"] = comm_channel_tree_cnt_df["nTrees1"] + (comm_channel_tree_cnt_df["numChannels"] % 2)
    comm_channel_tree_cnt_df = comm_channel_tree_cnt_df.set_index(["commId"])

    tree_cols = [
        "parent_node_id",
        "child_0_node_id",
        "child_1_node_id",
        "parent_child_type"
    ]

    for c in tree_cols:
        if c not in tree_ingress_egress_df.columns:
            tree_ingress_egress_df[c] = -1

    for line_idx in tree_ingress_egress_df.index:
        ## First update within each switch group
        row = tree_ingress_egress_df.loc[line_idx]
        comm_id = row['commId']
        channel_id = row['channelId']
        # tree_0_id = channel_id // 2
        nTrees0 = comm_channel_tree_cnt_df.loc[comm_id]["nTrees0"]
        use_Tree0 = 1 if channel_id < nTrees0 else 0
        tree_0_id = channel_id if use_Tree0 else channel_id - nTrees0
        node_id = row['nodeId']

        node_id_in_comm_switch_group = comm_nodes_groups_1.loc[(comm_id, node_id)]["nodeId_in_comm_switch_group"]
        num_nodes_in_comm_switch_group = comm_nodes_groups_1.loc[(comm_id, node_id)]["num_nodes_in_comm_switch_group"]
        comm_switch_group_id = comm_nodes_groups_1.loc[(comm_id, node_id)]["comm_switch_groupId"]
        root_node_id_in_comm_switch_group = tree_0_id % num_nodes_in_comm_switch_group
        node_id_in_comm_switch_group_shifted = commSwitchGroupNodeId_to_shiftedNodeId(node_id_in_comm_switch_group, root_node_id_in_comm_switch_group, num_nodes_in_comm_switch_group)

        parent_0_node_id_shifted, child_0_0_node_id_shifted, child_0_1_node_id_shifted, parentChildType0, parent_1_node_id_shifted, child_1_0_node_id_shifted, child_1_1_node_id_shifted, parentChildType1 = getDtree(num_nodes_in_comm_switch_group, node_id_in_comm_switch_group_shifted)
        parent_0_node_id_in_comm_switch_group = shiftedNodeId_to_commSwitchGroupNodeId(parent_0_node_id_shifted, root_node_id_in_comm_switch_group, num_nodes_in_comm_switch_group)
        child_0_0_node_id_in_comm_switch_group = shiftedNodeId_to_commSwitchGroupNodeId(child_0_0_node_id_shifted, root_node_id_in_comm_switch_group, num_nodes_in_comm_switch_group)
        child_0_1_node_id_in_comm_switch_group = shiftedNodeId_to_commSwitchGroupNodeId(child_0_1_node_id_shifted, root_node_id_in_comm_switch_group, num_nodes_in_comm_switch_group)
        parent_1_node_id_in_comm_switch_group = shiftedNodeId_to_commSwitchGroupNodeId(parent_1_node_id_shifted, root_node_id_in_comm_switch_group, num_nodes_in_comm_switch_group)
        child_1_0_node_id_in_comm_switch_group = shiftedNodeId_to_commSwitchGroupNodeId(child_1_0_node_id_shifted, root_node_id_in_comm_switch_group, num_nodes_in_comm_switch_group)
        child_1_1_node_id_in_comm_switch_group = shiftedNodeId_to_commSwitchGroupNodeId(child_1_1_node_id_shifted, root_node_id_in_comm_switch_group, num_nodes_in_comm_switch_group)

        parent_0_node_id = commSwitchGroupNodeId_to_globalNodeId(parent_0_node_id_in_comm_switch_group, comm_nodes_groups_2, comm_id, comm_switch_group_id)
        child_0_0_node_id = commSwitchGroupNodeId_to_globalNodeId(child_0_0_node_id_in_comm_switch_group, comm_nodes_groups_2, comm_id, comm_switch_group_id)
        child_0_1_node_id = commSwitchGroupNodeId_to_globalNodeId(child_0_1_node_id_in_comm_switch_group, comm_nodes_groups_2, comm_id, comm_switch_group_id)
        parent_1_node_id = commSwitchGroupNodeId_to_globalNodeId(parent_1_node_id_in_comm_switch_group, comm_nodes_groups_2, comm_id, comm_switch_group_id)
        child_1_0_node_id = commSwitchGroupNodeId_to_globalNodeId(child_1_0_node_id_in_comm_switch_group, comm_nodes_groups_2, comm_id, comm_switch_group_id)
        child_1_1_node_id = commSwitchGroupNodeId_to_globalNodeId(child_1_1_node_id_in_comm_switch_group, comm_nodes_groups_2, comm_id, comm_switch_group_id)

        parent_node_id = parent_0_node_id if use_Tree0 else parent_1_node_id
        child_0_node_id = child_0_0_node_id if use_Tree0 else child_1_0_node_id
        child_1_node_id = child_0_1_node_id if use_Tree0 else child_1_1_node_id
        parent_child_type = parentChildType0 if use_Tree0 else parentChildType1  ## parent_child_type: 0 means child_0 of the parent, 1 means child_1 of the parent, None means no parent
        parent_child_type = -1 if parent_child_type == None else parent_child_type

        ## Then update across switch groups if there are more than one switch group
        if parent_node_id == -1:
            assert parent_child_type == -1 and child_0_node_id == -1
            if (comm_id, comm_switch_group_id + 1, 0) in comm_nodes_groups_2.index:
                row_up = comm_nodes_groups_2.loc[(comm_id, comm_switch_group_id + 1, 0)]
                num_nodes_in_comm_switch_group_up = row_up['num_nodes_in_comm_switch_group']
                r_0, r_1 = getDtreeRoots(num_nodes_in_comm_switch_group_up)
                root_node_id_in_comm_switch_group_up_shifted = r_0 if use_Tree0 else r_1
                root_node_id_in_comm_switch_group_up_ref = tree_0_id % num_nodes_in_comm_switch_group_up
                root_node_id_in_comm_switch_group_up = shiftedNodeId_to_commSwitchGroupNodeId(root_node_id_in_comm_switch_group_up_shifted, root_node_id_in_comm_switch_group_up_ref, num_nodes_in_comm_switch_group_up)
                root_node_id_up = commSwitchGroupNodeId_to_globalNodeId(root_node_id_in_comm_switch_group_up, comm_nodes_groups_2, comm_id, comm_switch_group_id + 1)
                parent_node_id = root_node_id_up
                parent_child_type = 0
            
            if (comm_id, comm_switch_group_id - 1, 0) in comm_nodes_groups_2.index:
                row_down = comm_nodes_groups_2.loc[(comm_id, comm_switch_group_id - 1, 0)]
                num_nodes_in_comm_switch_group_down = row_down['num_nodes_in_comm_switch_group']
                r_0, r_1 = getDtreeRoots(num_nodes_in_comm_switch_group_down)
                root_node_id_in_comm_switch_group_down_shifted = r_0 if use_Tree0 else r_1
                root_node_id_in_comm_switch_group_down_ref = tree_0_id % num_nodes_in_comm_switch_group_down
                root_node_id_in_comm_switch_group_down = shiftedNodeId_to_commSwitchGroupNodeId(root_node_id_in_comm_switch_group_down_shifted, root_node_id_in_comm_switch_group_down_ref, num_nodes_in_comm_switch_group_down)
                root_node_id_down = commSwitchGroupNodeId_to_globalNodeId(root_node_id_in_comm_switch_group_down, comm_nodes_groups_2, comm_id, comm_switch_group_id - 1)
                child_0_node_id = root_node_id_down

        ## write back
        tree_ingress_egress_df.loc[line_idx, tree_cols] = [
            parent_node_id,
            child_0_node_id,
            child_1_node_id,
            parent_child_type
        ]

    return tree_ingress_egress_df

def update_comm_tree_info(comm_tree_info: pd.DataFrame, tree_ingress_egress_df: pd.DataFrame) -> pd.DataFrame:
    for line_idx in tree_ingress_egress_df.index:
        row = tree_ingress_egress_df.loc[line_idx]
        comm_id = row['commId']
        channel_id = row['channelId']
        node_id = row['nodeId']

        egressRank = row['egressRank']
        ingressMaybeRank_0 = row['ingressMaybeRank_0']
        ingressMaybeRank_1 = row['ingressMaybeRank_1']

        parent_node_id = row['parent_node_id']
        child_0_node_id = row['child_0_node_id']
        child_1_node_id = row['child_1_node_id']
        parent_child_type = row['parent_child_type']

        # print(f"{parent_node_id}, {type(parent_node_id)}")
        # print(f"{child_0_node_id}, {type(child_0_node_id)}")
        # print(f"{child_1_node_id}, {type(child_1_node_id)}")

        if parent_node_id != -1:
            mask = (
                (comm_tree_info['commId'] == comm_id) &
                (comm_tree_info['channelId'] == channel_id) &
                (comm_tree_info['nodeId'] == node_id) &
                (comm_tree_info['myRank'] == egressRank)
            )
            comm_tree_info.loc[mask, 'parentRank'] = tree_ingress_egress_df.loc[
                                                        (tree_ingress_egress_df['commId'] == comm_id) &
                                                        (tree_ingress_egress_df['channelId'] == channel_id) &
                                                        (tree_ingress_egress_df['nodeId'] == parent_node_id)
                                                    ].iloc[0]['ingressMaybeRank_0']
        else:
            mask = (
                (comm_tree_info['commId'] == comm_id) &
                (comm_tree_info['channelId'] == channel_id) &
                (comm_tree_info['nodeId'] == node_id) &
                (comm_tree_info['myRank'] == egressRank)
            )
            comm_tree_info.loc[mask, 'parentRank'] = -1
        
        if child_0_node_id != -1:
            external_child_0_rank = tree_ingress_egress_df.loc[
                                            (tree_ingress_egress_df['commId'] == comm_id) &
                                            (tree_ingress_egress_df['channelId'] == channel_id) &
                                            (tree_ingress_egress_df['nodeId'] == child_0_node_id)
                                        ].iloc[0]['egressRank']
        else:
            external_child_0_rank = -1

        if child_1_node_id != -1:
            external_child_1_rank = tree_ingress_egress_df.loc[
                                            (tree_ingress_egress_df['commId'] == comm_id) &
                                            (tree_ingress_egress_df['channelId'] == channel_id) &
                                            (tree_ingress_egress_df['nodeId'] == child_1_node_id)
                                        ].iloc[0]['egressRank']
        else:
            external_child_1_rank = -1

        mask = (
                (comm_tree_info['commId'] == comm_id) &
                (comm_tree_info['channelId'] == channel_id) &
                (comm_tree_info['nodeId'] == node_id) &
                (comm_tree_info['myRank'] == ingressMaybeRank_0)
            )

        internal_child_rank = -1
        maybe_internal_child_rank = comm_tree_info.loc[mask, 'child1Rank'].iloc[0]
        if maybe_internal_child_rank != -1:
            # print(f"line_idx: {line_idx}, comm_id: {comm_id}, channel_id: {channel_id}, maybe_internal_child_rank: {maybe_internal_child_rank}")
            maybe_internal_child_node = comm_tree_info.loc[
                                            (comm_tree_info['commId'] == comm_id) &
                                            (comm_tree_info['channelId'] == channel_id) &
                                            (comm_tree_info['myRank'] == maybe_internal_child_rank)
                                        ].iloc[0]['nodeId']
            if maybe_internal_child_node == node_id:
                internal_child_rank = maybe_internal_child_rank
        
        
        children_list = []
        for child in [internal_child_rank, external_child_0_rank, external_child_1_rank]:
            if child != -1: children_list.append(child)
        while len(children_list) < 3:
            children_list.append(-1)
        
        comm_tree_info.loc[mask, 'child1Rank'] = children_list[0]
        comm_tree_info.loc[mask, 'child2Rank'] = children_list[1]
        comm_tree_info.loc[mask, 'child3Rank'] = children_list[2]

    return comm_tree_info

