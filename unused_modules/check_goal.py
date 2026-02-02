import argparse
from typing import Dict, List
from collections import defaultdict
import re


def process_goal_file(input_file):
    nodes: Dict[int, Dict[int, str]] = {}
    edges: Dict[int, Dict[int, List[int]]] = {}
    with open(input_file, 'r') as f:
        f.readline() # skip the first line
        curr_nodes = None
        curr_edges = None
        curr_rank = -1
        for line in f:
            line = line.strip()
            if line.startswith("rank"):
                curr_rank = int(line.split()[1])
                curr_nodes = {}
                curr_edges = defaultdict(list)
            elif line.startswith("l"):
                if (match := re.match(r"l(\d+): (.*)", line)):
                    node_id = int(match.group(1))
                    operation = match.group(2)
                    curr_nodes[node_id] = operation
                elif (match := re.match(r"l(\d+) i?requires l(\d+)", line)):
                    src_id = int(match.group(2))
                    dst_id = int(match.group(1))
                    curr_edges[src_id].append(dst_id)
            elif line == "}":
                nodes[curr_rank] = curr_nodes
                edges[curr_rank] = curr_edges
                curr_nodes = None
                curr_edges = None
                curr_rank = -1
    return nodes, edges

def all_nodes_connected(nodes: Dict[int, str], edges: Dict[int, List[int]]):
    # make sure that all nodes are connected within one single graph
    # assume node 0 is the starting point
    all_nodes = set(nodes.keys())
    visited = set()
    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        for neighbor in edges.get(node, []):
            dfs(neighbor)
    dfs(0)
    return visited == all_nodes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a .goal file to extract nodes and edges.")
    parser.add_argument("input_file", type=str, help="Path to the input .goal file")
    args = parser.parse_args()

    nodes, edges = process_goal_file(args.input_file)
    for rank, rank_edges in edges.items():
        if not all_nodes_connected(nodes[rank], rank_edges):
            print(f"Rank {rank} has disconnected nodes.")
        else:
            print(f"Rank {rank} is fully connected.")