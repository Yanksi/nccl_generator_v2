from __future__ import annotations
from abc import ABC, abstractmethod
from collections import deque
import pathlib

class Graph:
    def __init__(self):
        self.in_edges = {}
        self.out_edges = {}
        self.nodes = {}
        self.start_nodes = set()
    
    def add_edge(self, from_node, to_node):
        if from_node not in self.nodes or to_node not in self.nodes:
            raise ValueError("Both nodes must be added to the graph before adding an edge.")
        self.out_edges.setdefault(from_node, []).append(to_node)
        self.in_edges.setdefault(to_node, []).append(from_node)
        if to_node in self.start_nodes:
            self.start_nodes.remove(to_node)
    
    def add_node(self, node_id, node_content):
        if node_id in self.nodes:
            raise ValueError(f"Node {node_id} already exists in the graph.")
        self.nodes[node_id] = node_content
        self.start_nodes.add(node_id)
    
    def __str__(self):
        nodes_str = "\n".join([f"{node_id}: {content}" for node_id, content in self.nodes.items()])
        edges_str = "\n".join([f"{from_node} -> {to_node}" for from_node, to_nodes in self.out_edges.items() for to_node in to_nodes])
        starting_nodes_str = ", ".join(self.start_nodes)
        return f"Nodes:\n{nodes_str}\n\nEdges:\n{edges_str}\n\nStarting Nodes:\n{starting_nodes_str}\n"
    
    def __repr__(self):
        return self.__str__()
    
    def __iter__(self):
        return GraphIterator(self)


class IterationNode():
    def __init__(self, graph_iter: GraphIterator, node_id: str, content: any):
        self.node_id = node_id
        self.iterator = graph_iter
        self.content = content
    
    def start(self):
        self.iterator.start(self.node_id)
    
    def finish(self):
        self.iterator.finish(self.node_id)

class GraphIterator():
    def __init__(self, graph: Graph):
        self.graph = graph
        self.available_nodes = graph.start_nodes.copy()
        self.visiting_nodes = set()
        self.visited_nodes = set()
    
    def start(self, node_id):
        if node_id not in self.available_nodes:
            raise ValueError(f"Node {node_id} is not available to start iteration.")
        self.available_nodes.remove(node_id)
        self.visiting_nodes.add(node_id)
    
    def finish(self, node_id):
        if node_id not in self.visiting_nodes:
            raise ValueError(f"Node {node_id} is not currently being visited.")
        self.visiting_nodes.remove(node_id)
        self.visited_nodes.add(node_id)
        
        for neighbor in self.graph.out_edges.get(node_id, []):
            if all(pred in self.visited_nodes for pred in self.graph.in_edges.get(neighbor, [])):
                self.available_nodes.add(neighbor)

    def __next__(self):
        if not self.available_nodes:
            raise StopIteration
        node = self.available_nodes.pop()
        return IterationNode(self, node, self.graph.nodes[node])
    

def goal_to_graph(goal_file_path: str) -> Graph:
    results = {}
    with open(goal_file_path, 'r') as f:
        n_ranks = int(f.readline().strip().split()[1])
        for r in range(n_ranks):
            curr_graph = Graph()
            curr_rank = None
            for line in f:
                if line.strip() == "}":
                    break 
                if line.strip().endswith("{"):
                    curr_rank = int(line.strip().split()[-2])
                    continue
                if ":" in line:
                    parts = line.strip().split(": ")
                    node_id = parts[0]
                    node_content = parts[1]
                    curr_graph.add_node(node_id, node_content)
                if "requires" in line:
                    parts = line.strip().split(" requires ")
                    from_node = parts[1]
                    to_node = parts[0]
                    curr_graph.add_edge(from_node, to_node)
            results[curr_rank] = curr_graph
    return results

if __name__ == "__main__":
    script_dir = pathlib.Path(__file__).parent
    graph = goal_to_graph(script_dir/"mix_edge.goal")
    print(graph)
