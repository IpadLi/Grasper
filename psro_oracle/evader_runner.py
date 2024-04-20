import math
import random

import numpy as np


class evader_runner:
    def __init__(self, game):
        self.game = game
        self.evader_path = self.game.attacker_path

        self.exit_node_list = self.game._graph.exit_node
        self.action_number = len(self.exit_node_list)
        self.avail_actions = [i for i in range(len(self.exit_node_list))]
        self.avail_actions_masked = []
        for i in range(self.action_number):
            if len(self.evader_path[self.exit_node_list[i]]) > 0:
                self.avail_actions_masked.append(i)
        self.q_table = np.array([random.uniform(-1, 1) + 1e-6 if len(self.evader_path[self.exit_node_list[i]]) > 0 else -99999 for i in range(self.action_number)])

    def get_strategy(self):
        q_value = [math.exp(i) for i in self.q_table]
        sum_value = sum(q_value)
        strategy = [q / sum_value for q in q_value]
        return strategy

    def get_action(self):
        strategy = self.get_strategy()
        action = np.random.choice(self.avail_actions, p=np.array(strategy).ravel())
        exit_node = self.exit_node_list[action]
        available_action = [i for i in range(len(self.evader_path[exit_node]))]
        idx = np.random.choice(available_action)
        return self.evader_path[exit_node][idx]

    def train(self, pursuer_policy_list, meta_probability, sample_number):
        for i in self.avail_actions:
            if len(self.evader_path[self.exit_node_list[i]]) > 0:
                reward = 0
                for _ in range(sample_number):
                    r_idx = np.random.choice(range(len(pursuer_policy_list)), p=meta_probability)
                    pursuer_policy = pursuer_policy_list[r_idx]
                    exit_node = self.exit_node_list[i]
                    available_action = [i for i in range(len(self.evader_path[exit_node]))]
                    idx = np.random.choice(available_action)
                    path = self.evader_path[exit_node][idx]
                    time_step = self.game.reset()
                    t = 0
                    while not time_step.last():
                        action = pursuer_policy.step(time_step, t)
                        time_step = self.game.step(path, action)
                        t += 1
                    reward += time_step.rewards[0]
                reward /= sample_number
                self.q_table[i] = reward

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
