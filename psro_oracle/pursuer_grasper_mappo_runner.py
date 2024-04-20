import numpy as np
import torch
from grasper_mappo.runner_shared.env_runner import EnvRunner
from grasper_mappo.envs.env import RL_Env
import dgl
from utils.graph_learning_utils import get_dgl_graph


class pursuer_grasper_mappo_runner:
    def __init__(self, mappo_args, args, game, pooled_node_emb, T):
        self.mappo_args = mappo_args
        self.args = args
        self.game = game
        self.pooled_node_emb = pooled_node_emb
        self.T = T

        self.env = RL_Env(game)
        self.env_runner = EnvRunner(self.env, mappo_args, args)

        if args.use_end_to_end:
            hgs = get_dgl_graph(game)
            hgs_batch = dgl.batch([hgs])
            hgs_batch = hgs_batch.to(args.device)
            Ts = torch.FloatTensor(np.array([T])).to(args.device)
            wa, ba, _, pooled_node_emb = self.env_runner.trainer.policy.actor.base.get_weight(hgs_batch, Ts)
            self.env_runner.trainer_ft.policy.actor.init_paras(wa, ba)
            self.pooled_node_emb = pooled_node_emb.numpy()
            wc, bc, _, _ = self.env_runner.trainer.policy.critic.base.get_weight(hgs_batch, Ts)
            self.env_runner.trainer_ft.policy.critic.init_paras(wc, bc)
        else:
            graph_embs = torch.FloatTensor(pooled_node_emb).to(args.device)
            Ts = torch.FloatTensor(T).to(args.device)
            wa, ba = self.env_runner.trainer.policy.actor.base.get_weight(graph_embs.unsqueeze(0), Ts.unsqueeze(0))
            self.env_runner.trainer_ft.policy.actor.init_paras(wa, ba)
            wc, bc = self.env_runner.trainer.policy.critic.base.get_weight(graph_embs.unsqueeze(0), Ts.unsqueeze(0))
            self.env_runner.trainer_ft.policy.critic.init_paras(wc, bc)
        if self.args.use_emb_layer and self.args.load_pretrain_model:
            self.load_emb_layer()

    def load_emb_layer(self):
        self.env_runner.trainer_ft.policy.actor.init_emb_layer(
            self.env_runner.trainer.policy.actor.base.node_idx_emb_layer.state_dict(),
            self.env_runner.trainer.policy.actor.base.time_idx_emb_layer.state_dict(),
            self.env_runner.trainer.policy.actor.base.agent_id_emb_layer.state_dict())
        self.env_runner.trainer_ft.policy.critic.init_emb_layer(
            self.env_runner.trainer.policy.critic.base.node_idx_emb_layer.state_dict(),
            self.env_runner.trainer.policy.critic.base.time_idx_emb_layer.state_dict())

    def step(self, time_step, t):
        obs = np.array(time_step.observations["observations"])
        shared_obs = np.array([time_step.observations["info_state"] for _ in range(self.env.defender_num)])
        obs = np.concatenate((shared_obs, np.expand_dims(obs[:, -1], 1)), axis=1)  # include own id
        if self.args.use_emb_layer:
            obs = torch.LongTensor(obs).to(self.args.device)
        else:
            obs = torch.FloatTensor(obs).to(self.args.device)
        actions = self.env_runner.trainer_ft.policy.act(obs, self.pooled_node_emb, batch=True)
        return actions

    def train(self, runners_list, meta_strategy, train_number, train_num_per_ite=1):
        attacker_action_number = runners_list[0].action_number
        strategy = [0 for _ in range(attacker_action_number)]
        for i, runner in enumerate(runners_list):
            strategy_temp = [s * meta_strategy[i] for s in runner.get_strategy()]
            strategy = [strategy_temp[idx] + strategy[idx] for idx in range(len(strategy))]
        strategy = np.array(strategy)
        strategy /= strategy.sum()
        self.env_runner.env.initialize_attacker_strategy(strategy)

        for i in range(train_number):
            self.env_runner.trainer_ft.policy.actor.eval()
            self.env_runner.trainer_ft.policy.critic.eval()
            _ = self.env_runner.run_one_episode_ft(self.pooled_node_emb, use_act_supervisor=False)
            _ = self.env_runner.train_ft(train_num_per_ite)