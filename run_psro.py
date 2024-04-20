import os
import torch
import random
import numpy as np

from psro_oracle.psro_oracle import PSRO_Oracle
from grasper_mappo.config import get_mtl_model_results_dir as get_grasper_mappo_model
from mappo.config import get_mtl_model_results_dir as get_mappo_model
from psro_oracle.utils import get_result_dir
import pickle
import os.path as osp
from args_cfg import get_args


def psro_run(args):
    args.lr_pi = 1e-4
    args.lr_vf = 5e-4
    if args.graph_type == 'Grid_Graph':
        pth_game_test = f'data/related_files/zero_shot_test/grid_{args.row * args.column}_probability_{args.edge_probability}'
    elif args.graph_type == 'SG_Graph':
        pth_game_test = f'data/related_files/zero_shot_test/sg_graph_probability_{args.edge_probability}'
    elif args.graph_type == 'SY_Graph':
        pth_game_test = f'data/related_files/zero_shot_test/sy_graph'
    elif args.graph_type == 'SF_Graph':
        pth_game_test = f'data/related_files/zero_shot_test/sf_graph_{args.sf_sw_node_num}'
    else:
        raise ValueError("Unkown graph type")
    ind_game_pth = pth_game_test + '/ind_games'
    ood_game_pth = pth_game_test + '/ood_games'
    ind_game_file = osp.join(ind_game_pth, f"ind_games_dnum{args.num_defender}_enum{args.num_exit}_T{args.min_time_horizon}_{args.max_time_horizon}_mep{args.min_evader_pth_len}_thd{args.ind_thd_min}_{args.ind_thd_max}_num_test_games{args.num_test_games}.pik")
    ood_game_file = osp.join(ood_game_pth, f"ood_games_dnum{args.num_defender}_enum{args.num_exit}_T{args.min_time_horizon}_{args.max_time_horizon}_mep{args.min_evader_pth_len}_thd{args.ood_thd_min}_{args.ood_thd_max}_num_test_games{args.num_test_games}.pik")

    if args.ood_test:
        if os.path.exists(ood_game_file):
            test_games = pickle.load(open(ood_game_file, 'rb'))['ood_games']
        else:
            raise ValueError(f"Pickle file not found: {ood_game_file}")
    else:
        if os.path.exists(ind_game_file):
            test_games = pickle.load(open(ind_game_file, 'rb'))['ind_games']
        else:
            raise ValueError(f"Pickle file not found: {ind_game_file}")

    worst_case_utility_list, time_list = [], []
    for game_idx, game in enumerate(test_games):
        args.test_game_idx = game_idx
        if args.load_pretrain_model:
            if args.pursuer_runner_type == "grasper_mappo":
                args.actor_model = get_grasper_mappo_model(args, args.pretrain_model_iteration) + "_actor.pt"
                args.critic_model = get_grasper_mappo_model(args, args.pretrain_model_iteration) + "_critic.pt"
            elif args.pursuer_runner_type == "mappo":
                args.actor_model = get_mappo_model(args, args.pretrain_model_iteration) + "_actor.pt"
                args.critic_model = get_mappo_model(args, args.pretrain_model_iteration) + "_critic.pt"
        psro = PSRO_Oracle(args, game)
        psro.init()
        times, worst_case_utilities = psro.solve()
        time_list.append(times)
        worst_case_utility_list.append(worst_case_utilities)
        print(worst_case_utility_list[game_idx])
        pickle.dump({'time_list': time_list, 'worst_case_utility_list': worst_case_utility_list}, open(get_result_dir(args), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    device_id = [0, 1, 2, 3, 4, 5, 6, 7]
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"
    args = get_args()
    args.cuda = torch.cuda.is_available()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)
    psro_run(args)
