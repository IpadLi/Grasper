import copy
import torch
import numpy as np

from game_env.game_state import State
from open_spiel.python.rl_environment import TimeStep, StepType
from game_env.utils import norm_adj

class Game(object):
    def __init__(self,
                 graph,
                 time_horizon,
                 max_time_horizon,
                 initial_locations,
                 attacker_path_list,
                 node_feat_dim):
        self._graph = graph
        self._time_horizon = time_horizon
        self._max_time_horizon = max_time_horizon
        self._initial_location = copy.deepcopy(initial_locations)
        self._defender_num = len(initial_locations[1])
        self.attacker_path = attacker_path_list
        self._node_feat_dim = node_feat_dim
        self.agent_num = self._defender_num + 1

        self._state = None
        self._should_reset = True

        self.defender_mix_action = len(graph.change_state[0])
        self.defender_state_representation_size = 1 + self._defender_num + 1 # evader loc + all defender locs + time step
        self.observation_size = 1 + 1 + 1 + 1 # evader loc + own loc + time step + own id

    def set_initial_location(self, initial_location):
        self._initial_location = copy.deepcopy(initial_location)

    def get_initial_location(self):
        return self._initial_location

    def get_exit_node_location(self):
        return self._graph.exit_node

    def condition_to_str(self):
        return f"T{self._time_horizon}_loc{self._initial_location}_exit{self._graph.exit_node}"

    def get_graph_info(self, ret_adj=False, normalize_adj=True):
        MAX_DEGREES = 30
        dgre = np.array(self._graph.graph.degree())[:,1]
        dgre[dgre > MAX_DEGREES] = MAX_DEGREES
        dgre_one_hot = np.eye(MAX_DEGREES + 1)[dgre]
        feat = np.zeros((self._graph.total_node_number, self._node_feat_dim))
        for node in self._graph.exit_node:
            feat[node - 1, 0] = 1
        feat[self._initial_location[0] - 1, 1] = 1
        for node in self._initial_location[1]:
            feat[node - 1, 2] += 1
        feat = np.concatenate((feat, dgre_one_hot), axis=1)
        if ret_adj:
            if normalize_adj:
                return feat, norm_adj(np.asarray(self._graph.adj_matrix))
            else:
                return feat, np.asarray(self._graph.adj_matrix)
        else:
            return feat

    def reset(self):
        initial_location = copy.deepcopy(self._initial_location)
        self._state = State(graph=self._graph,
                            time_horizon=self._time_horizon,
                            max_time_horizon=self._max_time_horizon,
                            initial_locations=initial_location)
        return self.get_time_step()

    def step(self, attacker_path, defender_action):
        if self._should_reset:
            self.reset()
        else:
            self._state.apply_action(attacker_path, defender_action)
        return self.get_time_step()

    def get_state_list(self):
        return self._state.get_current_state()

    def get_state_tensor(self):
        return torch.Tensor(self.get_state_list())

    def get_time_step(self):
        observations = {"info_state": [], "legal_actions": [], "observations": []}
        step_type = StepType.LAST if self._state.is_terminal() else StepType.MID
        self._should_reset = step_type == StepType.LAST
        rewards = self._state.rewards()
        observations["info_state"] = self._state.get_current_state()
        observations["legal_actions"] = self._state.legal_actions()
        observations["observations"] = self._state.get_observation()
        return TimeStep(observations=observations, rewards=rewards, discounts=[1.0], step_type=step_type)
