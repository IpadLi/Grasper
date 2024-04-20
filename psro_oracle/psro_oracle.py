import time
import random
import sys
import numpy as np
import torch

from psro_oracle.evader_runner import evader_runner
from psro_oracle.prd_solver import projected_replicator_dynamics
from psro_oracle.pursuer_grasper_mappo_runner import pursuer_grasper_mappo_runner
from psro_oracle.pursuer_mappo_runner import pursuer_mappo_runner
from psro_oracle.pursuer_random_runner import pursuer_random_runner
from grasper_mappo.config import get_config

from graph_learning.encoder import PreModel
from utils.graph_learning_utils import get_dgl_graph

class PSRO_Oracle:
    def __init__(self, args, game):
        if args.pursuer_runner_type == 'mappo':
            self.setup_str = f"Emb Layer: {args.use_emb_layer}, Augment: {args.use_augmentation}"
        elif args.pursuer_runner_type == 'grasper_mappo':
            if args.use_end_to_end:
                args.use_emb_layer = True
            self.setup_str = f"End2End: {args.use_end_to_end}, Emb Layer: {args.use_emb_layer}, Augment: {args.use_augmentation}"
        feat = game.get_graph_info()
        if args.graph_type == 'Grid_Graph':
            args.node_num = args.row_max_for_state_emb * args.column_max_for_state_emb  # enable varied grid size
        else:
            args.node_num = feat.shape[0]
        args.feat_dim = feat.shape[1]
        args.defender_num = game._defender_num

        self.args = args
        self.game = game
        self._num_players = 2
        self.seed = args.seed
        self.eval_episode = args.eval_episode
        self.num_psro_iteration = args.psro_iteration
        self.train_evader_episode_number = args.train_evader_number
        self.train_pursuer_episode_number = args.train_pursuer_number
        self.pursuer_runner_type = args.pursuer_runner_type
        self.meta_solver = projected_replicator_dynamics

        self.pursuer_runners_list = []
        self.evader_runners_list = []
        self.meta_games = [np.array([[]], dtype=np.float32), np.array([[]], dtype=np.float32)]
        self.meta_strategies = [np.array([], dtype=np.float32), np.array([], dtype=np.float32)]

        parser = get_config()
        self.mappo_args = parser.parse_known_args(sys.argv[1:])[0]
        self.mappo_args.num_defender = game._defender_num
        self.mappo_args.device = args.device
        self.mappo_args.num_task = args.num_task
        self.mappo_args.num_sample = args.num_sample
        self.mappo_args.hidden_size = args.hidden_size
        self.mappo_args.num_iterations = args.num_iterations
        self.mappo_args.act_sup_coef_min = args.act_sup_coef_min
        self.mappo_args.act_sup_coef_max = args.act_sup_coef_max
        self.mappo_args.act_sup_coef_decay = args.act_sup_coef_decay

        if args.pursuer_runner_type == 'grasper_mappo' and args.use_end_to_end:
            self.pooled_node_emb = None
        else:
            graph_emb_model = PreModel(feat.shape[1], args.gnn_hidden_dim, args.gnn_output_dim, args.gnn_num_layer, args.gnn_dropout)
            graph_emb_model.to(args.device)
            if args.load_graph_emb_model_in_psro:
                print('Load pretrained graph model ...')
                pretrain_graph_model = f"data/pretrain_models/graph_learning/checkpoint_epoch{args.max_epoch}_type_{args.graph_type}_"\
                                       f"ep{args.edge_probability}_gp{args.pool_size}_layer{args.gnn_num_layer}_hidden{args.gnn_hidden_dim}_"\
                                       f"out{args.gnn_output_dim}_dnum{args.num_defender}_enum{args.num_exit}_mep{args.min_evader_pth_len}.pt"
                print(pretrain_graph_model)
                graph_emb_model.load(torch.load(pretrain_graph_model))
            graph_emb_model.eval()
            with torch.no_grad():
                hg = get_dgl_graph(game)
                hg = hg.to(args.device)
                feat = hg.ndata["attr"]
                _, pooled_node_emb = graph_emb_model.embed(hg, feat)
                self.pooled_node_emb = pooled_node_emb.cpu().numpy()
        self.T = np.array([0] * args.max_time_horizon_for_state_emb)
        self.T[self.game._time_horizon] = 1

    def get_runner(self, player_id):
        if player_id == 0:
            return evader_runner(self.game)
        if player_id == 1:
            if self.pursuer_runner_type == 'grasper_mappo':
                return pursuer_grasper_mappo_runner(self.mappo_args, self.args, self.game, self.pooled_node_emb, self.T)
            elif self.pursuer_runner_type == 'mappo':
                return pursuer_mappo_runner(self.mappo_args, self.args, self.game, self.pooled_node_emb, self.T)
            elif self.pursuer_runner_type == 'random':
                return pursuer_random_runner(self.mappo_args, self.args, self.game, self.pooled_node_emb, self.T)

    def init(self):
        self.pursuer_runners_list = []
        self.evader_runners_list = []
        self.meta_games = [np.array([[]], dtype=np.float32), np.array([[]], dtype=np.float32)]
        self.meta_strategies = [np.array([], dtype=np.float32), np.array([], dtype=np.float32)]

        evader_agent = self.get_runner(0)
        pursuer_agent = self.get_runner(1)
        evaluate_reward = self.evaluate([evader_agent, pursuer_agent])
        self.pursuer_runners_list.append(pursuer_agent)
        self.evader_runners_list.append(evader_agent)
        r = len(self.evader_runners_list)
        c = len(self.pursuer_runners_list)
        self.meta_games = [np.full([r, c], fill_value=evaluate_reward[0]),
                           np.full([r, c], fill_value=evaluate_reward[1])]
        self.meta_strategies = [np.array([1.0]), np.array([1.0])]

    def one_iteration(self, i):
        print('\n********************')
        print('Game idx:{}, Alg:{}, {}, Act Supervisor:{}, Map:{}, Pursuer:{}, Evader:{}, PSRO iter: {}/{}'
              .format(self.args.test_game_idx, self.pursuer_runner_type, self.setup_str, self.args.use_act_supervisor,
                      self.game._graph.total_node_number, self.train_pursuer_episode_number,
                      self.train_evader_episode_number,
                      i + 1, self.num_psro_iteration))
        print("add new runners...")
        worst_case_utility = self.add_new_runner()
        return worst_case_utility

    def solve(self):
        time_list = []
        worst_case_utility_list = []
        start_time = time.time()
        for i in range(self.num_psro_iteration):
            worst_case_utility = self.one_iteration(i)
            worst_case_utility_list.append(-worst_case_utility)
            time_list.append(time.time() - start_time)
            print("worst case utility for pursuer:", worst_case_utility_list)
            print("update meta game...")
            self.update_meta_game()
            print("compute meta distribution...")
            self.compute_meta_distribution()
        return time_list, worst_case_utility_list

    def add_evader_runner(self):
        evader_agent = self.get_runner(0)
        start_time = time.time()
        evader_agent.train(self.pursuer_runners_list, self.meta_strategies[1], self.train_evader_episode_number)
        best_0 = evader_agent.q_table[arg_max(evader_agent.q_table)]
        print("evader BR computation, run time: {:.6f}, BR value: {:.5f}".format(time.time() - start_time, best_0))
        return best_0, evader_agent

    def add_pursuer_runner(self):
        start_time = time.time()
        pursuer_agent = self.get_runner(1)
        pursuer_agent.train(self.evader_runners_list, self.meta_strategies[0], self.train_pursuer_episode_number, train_num_per_ite=10)
        print("pursuer BR computation, run time: {:.6f}".format(time.time() - start_time))
        return pursuer_agent

    def add_new_runner(self):
        best_0, evader_agent = self.add_evader_runner()
        pursuer_agent = self.add_pursuer_runner()
        self.evader_runners_list.append(evader_agent)
        self.pursuer_runners_list.append(pursuer_agent)
        return best_0

    def update_meta_game(self):
        r = len(self.evader_runners_list)
        c = len(self.pursuer_runners_list)
        meta_games = [np.full([r, c], fill_value=np.nan),
                      np.full([r, c], fill_value=np.nan)]
        (o_r, o_c) = self.meta_games[0].shape
        for i in [0, 1]:
            for t_r in range(o_r):
                for t_c in range(o_c):
                    meta_games[i][t_r][t_c] = self.meta_games[i][t_r][t_c]
        for t_r in range(r):
            for t_c in range(c):
                if np.isnan(meta_games[0][t_r][t_c]):
                    evaluate_reward = self.evaluate([self.evader_runners_list[t_r], self.pursuer_runners_list[t_c]])
                    meta_games[0][t_r][t_c] = evaluate_reward[0]
                    meta_games[1][t_r][t_c] = evaluate_reward[1]
        self.meta_games = meta_games

    def compute_meta_distribution(self):
        self.meta_strategies = self.meta_solver(self.meta_games)

    def evaluate(self, runner_list):
        totals = np.zeros(self._num_players)
        for _ in range(self.eval_episode):
            attacker_path = runner_list[0].get_action()
            time_step = self.game.reset()
            t = 0
            while not time_step.last():
                action = runner_list[1].step(time_step, t)
                time_step = self.game.step(attacker_path, action)
                t += 1
            totals += np.array(time_step.rewards).reshape(-1)
        return totals / self.eval_episode


def arg_max(state_action):
    max_index_list = []
    max_value = state_action[0]
    for index, value in enumerate(state_action):
        if value > max_value:
            max_index_list.clear()
            max_value = value
            max_index_list.append(index)
        elif value == max_value:
            max_index_list.append(index)
    return random.choice(max_index_list)
