import networkx as nx
import numpy as np
import os
import os.path as osp
import random

from game_env.graph.base_graph import graph_base


class SF_Graph(graph_base):
    def __init__(self, graph_type=None, exit_node=None, edge_probability=1.0, node_number=300, seed_to_generate_graph=100):

        self.seed_to_generate_graph = seed_to_generate_graph
        random.seed(seed_to_generate_graph)
        np.random.seed(seed_to_generate_graph)
        self.total_node_number = node_number
        self.type = graph_type
        self.edge_probability = edge_probability

        super().__init__(exit_node)

    def build_graph(self):
        save_path = 'game_env/graph/sf'
        if not osp.exists(save_path):
            os.makedirs(save_path)
        sf_graph_file = osp.join(save_path, f'sf_graph_node_num{self.total_node_number}.gpickle')
        if os.path.exists(sf_graph_file):
            g = nx.read_gpickle(sf_graph_file)
        else:
            g = nx.barabasi_albert_graph(self.total_node_number, 2)
            if not nx.is_connected(g):
                raise ValueError(f'Graph is not connected. Change the seed {self.seed_to_generate_graph} to other numbers to generate again.')
            mapping = {node: node + 1 for node in g.nodes}
            g = nx.relabel_nodes(g, mapping)
            nx.write_gpickle(g, sf_graph_file)
        max_actions = max(list(dict(g.degree()).values())) + 1
        map_adjlist = nx.to_dict_of_lists(g)
        for node in map_adjlist:
            map_adjlist[node].append(node)
            map_adjlist[node].sort()
        total_node_number = len(map_adjlist)
        self.change_state = [[i for _ in range(max_actions)] for i in range(1, total_node_number + 1)]
        self.legal_action = [[] for _ in range(1, total_node_number + 1)]
        for i in map_adjlist.keys():
            adjlist = map_adjlist[i]
            for idx, j in enumerate(adjlist):
                self.change_state[i - 1][idx] = j
                self.legal_action[i - 1].append(idx)

        return g, total_node_number, self.change_state, self.legal_action