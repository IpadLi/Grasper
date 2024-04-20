import os
import numpy as np
import os.path as osp
import networkx as nx

from game_env.graph.base_graph import graph_base


class SY_Graph(graph_base):
    def __init__(self, graph_type=None, exit_node=None, edge_probability=1.0):

        self.type = graph_type
        self.edge_probability = edge_probability

        super().__init__(exit_node)

    def build_graph(self):
        save_path = 'game_env/graph/sy'
        if not osp.exists(save_path):
            os.makedirs(save_path)
        graph_file = osp.join(save_path, f'sy_graph.gpickle')
        if os.path.exists(graph_file):
            g = nx.read_gpickle(graph_file)
        else:
            adj_matrix = np.zeros((200, 200))
            with open("game_env/graph/sy/neighbor_node_list.txt", "r") as f:
                for line in f:
                    data = [a.strip() for a in line.split("|")]
                    print(data)
                    start = int(data[0].strip())
                    print(start)
                    if data[1] != "":
                        print(data[1])
                        for a in data[1].split(" "):
                            adj_matrix[start - 1, int(a.strip()) - 1] = 1.0
                            adj_matrix[int(a.strip()) - 1, start - 1] = 1.0
            g = nx.from_numpy_matrix(adj_matrix)
            if not nx.is_connected(g):
                raise ValueError(f'Scotlan Yard graph is not connected.')
            mapping = {node: node + 1 for node in g.nodes}
            g = nx.relabel_nodes(g, mapping)
            nx.write_gpickle(g, graph_file)
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