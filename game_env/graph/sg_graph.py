import networkx as nx
import random

from game_env.graph.base_graph import graph_base


class SG_Graph(graph_base):
    def __init__(self, graph_type=None, exit_node=None, edge_probability=1.0):

        self.type = graph_type
        self.edge_probability = edge_probability

        super().__init__(exit_node)

    def build_graph(self):
        g = nx.read_gpickle("game_env/graph/sg/sg.gpickle")
        max_actions = max(list(dict(g.degree()).values())) + 1
        if self.edge_probability < 1.0:
            while True:
                new_g = g.copy()
                for edge in list(g.edges):
                    rnd, deg_0, deg_1 = random.random(), g.degree[edge[0]], g.degree[edge[1]]
                    if rnd > self.edge_probability and deg_0 > 3 and deg_1 > 3:
                        new_g.remove_edge(*edge)
                if nx.is_connected(new_g):
                    break
            g = new_g.copy()
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