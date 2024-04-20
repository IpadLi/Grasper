import numpy as np
from grasper_mappo.envs.env import RL_Env

class pursuer_random_runner:
    def __init__(self, mappo_args, args, game, pooled_node_emb, T):
        self.mappo_args = mappo_args
        self.args = args
        self.game = game
        self.pooled_node_emb = pooled_node_emb
        self.T = T

        self.env = RL_Env(game)
        self.action_num = self.env.action_dim

    def step(self, runners_list, meta_strategy, train_number, train_num_per_ite=1):
        actions = [np.random.randint(0, self.action_num) for _ in range(self.game._defender_num)]
        return actions