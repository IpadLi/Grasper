import time
import random
from game_env.build_game import build_game
import numpy as np

def get_game(args, compute_path=True):
    graph_type = args.graph_type
    row = args.row
    column = args.column
    edge_probability = args.edge_probability
    assert args.min_time_horizon <= args.max_time_horizon
    if args.min_time_horizon < args.max_time_horizon:
        time_horizon = np.random.randint(args.min_time_horizon, args.max_time_horizon)
    else:
        time_horizon = args.max_time_horizon

    if graph_type == 'Grid_Graph':
        exit_node_candidates = [i + 1 for i in range(column)] + [(row - 1) * column + i + 1 for i in range(column)] + \
                               [i * column + 1 for i in range(1, row - 1)] + [i * column + column for i in range(1, row - 1)]
        exit_node = list(np.random.choice(exit_node_candidates, args.num_exit, replace=False))
        exit_node = sorted(exit_node)
        feasible_locations = np.random.permutation([i for i in range(1, row * column + 1) if i not in exit_node])
        initial_locations = [feasible_locations[0], list(np.random.choice(list(feasible_locations[1:]), args.num_defender))]
    elif graph_type == 'SG_Graph':
        max_node_num = 372
        node_list = list(np.arange(max_node_num) + 1)
        exit_node = list(np.random.choice(node_list, args.num_exit, replace=False))
        exit_node = sorted(exit_node)
        feasible_locations = [node for node in node_list if node not in exit_node]
        initial_locations = [feasible_locations[0], list(np.random.choice(list(feasible_locations[1:]), args.num_defender))]
    elif graph_type == 'SF_Graph':
        random.seed(int(time.time()))
        np.random.seed(int(time.time()))
        max_node_num = args.sf_sw_node_num
        node_list = list(np.arange(max_node_num) + 1)
        rnd_node_list = np.random.permutation(node_list)
        exit_node = rnd_node_list[:args.num_exit]
        exit_node = sorted(exit_node)
        feasible_locations = [node for node in node_list if node not in exit_node]
        rnd_feasible_locations = np.random.permutation(feasible_locations)
        initial_locations = [rnd_feasible_locations[0], list(rnd_feasible_locations[1:args.num_defender + 1])]
    elif graph_type == 'SY_Graph':
        random.seed(int(time.time()))
        np.random.seed(int(time.time()))
        max_node_num = 200
        candidate_start_nodes = [103, 112, 34, 155, 94, 117, 132, 53, 174, 198, 50, 91, 26, 29, 141, 13, 138, 197]
        node_list = [i+1 for i in range(max_node_num) if i+1 not in candidate_start_nodes]
        rnd_node_list = np.random.permutation(node_list)
        exit_node = rnd_node_list[:args.num_exit]
        exit_node = sorted(exit_node)
        rnd_feasible_locations = np.random.permutation(candidate_start_nodes)
        initial_locations = [rnd_feasible_locations[0], list(rnd_feasible_locations[1:args.num_defender + 1])]
    else:
        raise ValueError(f"Unsupported graph type {graph_type}.")

    node_feat_dim = args.node_feat_dim

    game \
        = build_game(graph_type, row, column, edge_probability, exit_node, initial_locations,
                     time_horizon, args.max_time_horizon, node_feat_dim, compute_path=compute_path,
                     sf_sw_node_number=args.sf_sw_node_num, seed_to_generate_graph=args.seed_to_generate_graph)

    return game
