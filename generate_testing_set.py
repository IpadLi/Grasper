import os
import sys
import torch
import os.path as osp
import pickle
import random
import numpy as np

sys.path.append(".")

from utils.game_config import get_game
from args_cfg import get_args
from grasper_mappo.config import get_mtl_model_results_dir
from psro_oracle.psro_oracle import PSRO_Oracle


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

    if args.graph_type == 'Grid_Graph':
        pth_game_pool = 'data/related_files/game_pool/grid_{}_probability_{}'.format(args.row * args.column, args.edge_probability)
        pth_game_test = f'data/related_files/zero_shot_test/grid_{args.row * args.column}_probability_{args.edge_probability}'
    elif args.graph_type == 'SG_Graph':
        pth_game_pool = 'data/related_files/game_pool/sg_graph_probability_{}'.format(args.edge_probability)
        pth_game_test = f'data/related_files/zero_shot_test/sg_graph_probability_{args.edge_probability}'
    elif args.graph_type == 'SY_Graph':
        pth_game_pool = 'data/related_files/game_pool/sy_graph'
        pth_game_test = f'data/related_files/zero_shot_test/sy_graph'
    elif args.graph_type == 'SF_Graph':
        pth_game_pool = 'data/related_files/game_pool/sf_graph_{}'.format(args.sf_sw_node_num)
        pth_game_test = f'data/related_files/zero_shot_test/sf_graph_{args.sf_sw_node_num}'
    else:
        raise ValueError("Unkown graph type")
    game_pool_file = osp.join(pth_game_pool, 'game_pool_size{}_dnum{}_enum{}_T{}_{}_mep{}.pik'.format(
        args.pool_size, args.num_defender, args.num_exit, args.min_time_horizon, args.max_time_horizon, args.min_evader_pth_len))
    ind_game_pth = pth_game_test + '/ind_games'
    ood_game_pth = pth_game_test + '/ood_games'
    if not osp.exists(ind_game_pth):
        os.makedirs(ind_game_pth)
    if not osp.exists(ood_game_pth):
        os.makedirs(ood_game_pth)
    grasper_actor = get_mtl_model_results_dir(args, args.pretrain_model_iteration) + "_actor.pt"
    grasper_critic = get_mtl_model_results_dir(args, args.pretrain_model_iteration) + "_critic.pt"

    ind_game_file = osp.join(ind_game_pth, f"ind_games_dnum{args.num_defender}_enum{args.num_exit}_T{args.min_time_horizon}_{args.max_time_horizon}_mep{args.min_evader_pth_len}_thd{args.ind_thd_min}_{args.ind_thd_max}_num_test_games{args.num_test_games}.pik")
    ood_game_file = osp.join(ood_game_pth, f"ood_games_dnum{args.num_defender}_enum{args.num_exit}_T{args.min_time_horizon}_{args.max_time_horizon}_mep{args.min_evader_pth_len}_thd{args.ood_thd_min}_{args.ood_thd_max}_num_test_games{args.num_test_games}.pik")

    if not osp.exists(ood_game_file):
        print('********** Generation of OOD games begin **********')
        training_set = pickle.load(open(game_pool_file, 'rb'))['game_pool']
        game_pool_str = [game.condition_to_str() for game in training_set]
        ood_games = []
        avg_u = 0
        for game_idx in range(args.num_test_games):
            while True:
                while True:
                    game = get_game(args)
                    path_length = np.array([len(i[0]) if len(i) > 0 else 0 for i in game.attacker_path.values()])
                    if sum(path_length > 0) > 0 and min(path_length[path_length > 0]) >= args.min_evader_pth_len and game.condition_to_str() not in game_pool_str:
                        break
                if args.load_pretrain_model:
                    args.actor_model = grasper_actor
                    args.critic_model = grasper_critic
                psro = PSRO_Oracle(args, game)
                psro.init()
                worst_case_utility, _ = psro.add_evader_runner()
                if args.ood_thd_min <= -worst_case_utility <= args.ood_thd_max:
                    print(f'OOD Game {game_idx} found. ', 'Avg Utility: ', (avg_u - worst_case_utility) / (game_idx + 1))
                    ood_games.append(game)
                    pickle.dump({'ood_games': ood_games}, open(ood_game_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
                    break
        print('********** Generation of OOD games end **********')

    if not osp.exists(ind_game_file):
        print('********** Generation of Ind games begin **********')
        training_set = pickle.load(open(game_pool_file, 'rb'))['game_pool']
        ind_games = []
        avg_u, cnt = 0, 0
        for game_idx, game in enumerate(training_set):
            args.test_game_idx = game_idx
            if args.load_pretrain_model:
                args.actor_model = grasper_actor
                args.critic_model = grasper_critic
            psro = PSRO_Oracle(args, game)
            psro.init()
            worst_case_utility, _ = psro.add_evader_runner()
            if args.ood_thd_min <= -worst_case_utility <= args.ood_thd_max:
                cnt += 1
                print(f'IND Game {game_idx} found. ', 'Avg Utility: ', (avg_u - worst_case_utility) / cnt)
                ind_games.append(game)
                pickle.dump({'ind_games': ind_games}, open(ind_game_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
            if cnt == args.num_test_games:
                break
        print('********** Generation of Ind games end **********')