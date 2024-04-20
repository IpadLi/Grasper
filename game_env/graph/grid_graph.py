import random
import numpy as np
import networkx as nx
from game_env.graph.base_graph import graph_base


class Grid_Graph(graph_base):
    def __init__(self, graph_type=None, row=3, column=3, edge_probability=1.0, exit_node=None):
        self.row = row
        self.column = column
        self.type = graph_type
        self.edge_probability = edge_probability

        super().__init__(exit_node)

    def build_graph(self):
        self.total_node_number = self.column * self.row
        while True:
            self.edges = []
            self.change_state = [[i, i, i, i, i] for i in range(1, self.total_node_number + 1)]
            self.legal_action = [[0] for _ in range(1, self.total_node_number + 1)]
            count = 1
            line = 1
            # exit nodes locates in the edge of grid
            flag = True if self.exit_node == [] else False
            if flag:
                self.exit_node += [i + 1 for i in range(self.column)]
                self.exit_node += [i + 1 for i in range(self.total_node_number - self.column, self.total_node_number)]
            for i in range(self.total_node_number):
                if i + self.column < self.total_node_number:
                    if count == 1 or count == self.column:
                        self.change_state[i][2] = i + self.column + 1
                        self.change_state[i + self.column][1] = i + 1
                        self.edges.append([i + 1, i + self.column + 1])
                        self.edges.append([i + self.column + 1, i + 1])
                    else:
                        if random.random() <= self.edge_probability:
                            self.change_state[i][2] = i + self.column + 1
                            self.change_state[i + self.column][1] = i + 1
                            self.edges.append([i + 1, i + self.column + 1])
                            self.edges.append([i + self.column + 1, i + 1])
                if count != self.column:
                    if line == 1 or line == self.row:
                        self.change_state[i + 1][3] = i + 1
                        self.change_state[i][4] = i + 2
                        self.edges.append([i + 1, i + 2])
                        self.edges.append([i + 2, i + 1])
                    else:
                        if random.random() <= self.edge_probability:
                            self.change_state[i + 1][3] = i + 1
                            self.change_state[i][4] = i + 2
                            self.edges.append([i + 1, i + 2])
                            self.edges.append([i + 2, i + 1])
                    count += 1
                else:
                    if flag:
                        self.exit_node.append(i + 1)
                        self.exit_node.append(i + 2)
                    count = 1
                    line += 1
                for j, a in enumerate(self.change_state[i]):
                    if a != i + 1:
                        self.legal_action[i].append(j)
            if flag:
                # process exit node
                self.exit_node = self.exit_node[:-1]
                self.exit_node = set(self.exit_node)
                self.exit_node = list(set(self.exit_node))
            self.graph = nx.DiGraph()
            for i in range(self.total_node_number):
                self.graph.add_node(i + 1)
            self.graph.add_edges_from(np.array(self.edges))
            undirected_graph = self.graph.to_undirected()
            if nx.is_connected(undirected_graph):
                break
        return self.graph, self.total_node_number, self.change_state, self.legal_action
