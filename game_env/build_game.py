from game_env.game_env import Game
from game_env.graph.grid_graph import Grid_Graph
from game_env.graph.sg_graph import SG_Graph
from game_env.graph.sf_graph import SF_Graph
from game_env.graph.sy_graph import SY_Graph

def build_game(graph_type, row, column, edge_probability, exit_node,
               initial_locations, time_horizon, max_time_horizon, node_feat_dim,
               compute_path=True, sf_sw_node_number=300, seed_to_generate_graph=100):

    if graph_type == "Grid_Graph":
        graph = Grid_Graph(graph_type=graph_type, row=row, column=column, edge_probability=edge_probability, exit_node=exit_node)
    elif graph_type == 'SG_Graph':
        graph = SG_Graph(graph_type=graph_type, exit_node=exit_node, edge_probability=edge_probability)
    elif graph_type == 'SY_Graph':
        graph = SY_Graph(graph_type=graph_type, exit_node=exit_node, edge_probability=edge_probability)
    elif graph_type == 'SF_Graph':
        graph = SF_Graph(graph_type=graph_type, exit_node=exit_node, edge_probability=edge_probability,
                         node_number=sf_sw_node_number, seed_to_generate_graph=seed_to_generate_graph)
    else:
        raise ValueError(f"Unknown graph {graph_type}.")

    if compute_path:
        attacker_path_list = graph.get_shortest_path(initial_locations[0], time_horizon)
        attacker_path = attacker_path_list[1]
    else:
        attacker_path = []

    game = Game(graph=graph, time_horizon=time_horizon, max_time_horizon=max_time_horizon,
                initial_locations=initial_locations, node_feat_dim=node_feat_dim,
                attacker_path_list=attacker_path)

    return game
