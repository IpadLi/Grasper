import random
import numpy as np


class RL_Env(object):
    def __init__(self, game):
        self.game = game
        self.defender_num = self.game._defender_num
        self.agent_num = self.game.agent_num
        self.obs_dim = self.game.observation_size
        self.share_obs_dim = self.game.defender_state_representation_size
        self.action_dim = self.game.defender_mix_action
        self.action_number = len(self.game.attacker_path)   # num of exit nodes or num of paths
        self.exit_node = self.game._graph.exit_node
        self.time_horizon = self.game._time_horizon
        self.attacker_path = None
        self.initialize_attacker_strategy()

    def reset_by_game(self, game):
        self.game = game
        self.action_number = len(self.game.attacker_path)  # num of exit nodes or num of paths
        self.exit_node = self.game._graph.exit_node
        self.time_horizon = self.game._time_horizon

    def initialize_attacker_strategy(self, attacker_strategy=None):
        if attacker_strategy is None:
            s = np.array([random.uniform(1, 10) if len(self.game.attacker_path[self.exit_node[i]]) > 0 else 0 for i in range(self.action_number)])
            s /= s.sum()
            self.attacker_strategy = s
        else:
            self.attacker_strategy = attacker_strategy

    def step(self, action):
        # action is a list, each for one defender
        if self.game._should_reset:
            done = True
            reward = [0]
            state = [0 for _ in range(self.share_obs_dim)]  # shape: 1 * state_size
            sub_agent_obs = [[0 for _ in range(self.obs_dim)] for _ in range(self.defender_num)]
            time_step = self.game.reset()
        else:
            time_step = self.game.step(self.attacker_path, action)
            done = True if time_step.last() else False
            reward = [time_step.rewards[1]]
            state = time_step.observations["info_state"]  # shape: 1 * state_size
            sub_agent_obs = time_step.observations["observations"]

        share_obs = [state for _ in range(self.defender_num)]
        sub_agent_reward = [reward for _ in range(self.defender_num)]
        sub_agent_done = [done for _ in range(self.defender_num)]

        return [np.array(share_obs), np.array(sub_agent_obs), np.array(sub_agent_reward), np.array(sub_agent_done), time_step]

    def reset(self):
        action_index = np.random.choice(range(self.action_number), p=self.attacker_strategy)
        exit_node = self.exit_node[action_index]
        available_action = [i for i in range(len(self.game.attacker_path[exit_node]))]
        idx = np.random.choice(available_action)
        self.attacker_path = self.game.attacker_path[exit_node][idx]
        time_step = self.game.reset()
        sub_agent_obs = time_step.observations["observations"]  # shape: n_agent * obs_size
        state = time_step.observations["info_state"]  # shape: 1 * state_size
        share_obs = [state for _ in range(self.defender_num)]  # shape: n_agent * state_size
        return [np.array(share_obs), np.array(sub_agent_obs), time_step]
