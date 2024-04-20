import numpy as np
import networkx as nx


class graph_base(object):
    def __init__(self, exit_node=None):
        self.exit_node = [] if exit_node is None else exit_node
        self.graph, self.total_node_number, self.change_state, self.legal_action = self.build_graph()
        self.graph = nx.DiGraph(self.graph)
        self.adj_matrix = nx.to_numpy_array(self.graph)
        self.edge_index = np.array([[e[0] - 1, e[1] - 1] for e in self.graph.edges]).transpose()

    def build_graph(self):
        return None, 0, None, []

    def get_next_node(self, current_node, action):
        return self.change_state[current_node - 1][action]

    def get_legal_action(self, current_node):
        return self.legal_action[current_node - 1]

    def get_shortest_path(self, node_number_start, length):
        path = []
        path_list = {}
        for j in range(len(self.exit_node)):
            path_temp = []
            if nx.has_path(self.graph, source=node_number_start, target=self.exit_node[j]):
                shortest_path = nx.all_shortest_paths(self.graph, source=node_number_start, target=self.exit_node[j])
                for p in shortest_path:
                    if len(p) > length + 1 or len(path_temp) >= 200:
                        break
                    else:
                        if list(set(p) & set(self.exit_node)) == [self.exit_node[j]]:
                            path_temp.append(p)
            path_list[self.exit_node[j]] = path_temp
            path += path_temp
        return path, path_list