import numpy as np
import torch
from mappo.runner_shared.env_runner import EnvRunner
from mappo.envs.env import RL_Env
from utils.utils import obs_query


class pursuer_mappo_runner:
    def __init__(self, mappo_args, args, game, pooled_node_emb, T):
        self.mappo_args = mappo_args
        self.args = args
        self.game = game
        if args.use_augmentation:
            self.pooled_node_embs = np.array([pooled_node_emb for _ in range(game._defender_num)])
            self.Ts = np.array([T for _ in range(game._defender_num)])
        else:
            self.pooled_node_embs = None
            self.Ts = None

        self.env = RL_Env(game)
        self.env_runner = EnvRunner(self.env, mappo_args, args, train_mode='finetune')

    def step(self, time_step, t):
        obs = np.array(time_step.observations["observations"])
        shared_obs = np.array([time_step.observations["info_state"] for _ in range(self.env.defender_num)])
        obs = np.concatenate((shared_obs, np.expand_dims(obs[:, -1], 1)), axis=1)  # include own id
        if self.args.use_emb_layer:
            obs = torch.LongTensor(obs).to(self.args.device)
        else:
            if self.args.use_augmentation:
                obs = obs_query( obs, self.args.max_time_horizon_for_state_emb, t)
            obs = torch.FloatTensor(obs).to(self.args.device)
        if self.pooled_node_embs is not None:
            pooled_node_embs = torch.FloatTensor(self.pooled_node_embs).to(self.args.device)
            Ts = torch.FloatTensor(self.Ts).to(self.args.device)
        else:
            pooled_node_embs, Ts = None, None
        actions = self.env_runner.trainer.policy.act(obs, pooled_node_embs, Ts, batch=True)
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
            self.env_runner.trainer.policy.actor.eval()
            self.env_runner.trainer.policy.critic.eval()
            _ = self.env_runner.run_one_episode(self.pooled_node_embs, self.Ts, use_act_supervisor=False)
            _ = self.env_runner.train(train_num_per_ite)