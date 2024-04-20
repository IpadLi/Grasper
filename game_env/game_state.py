import copy

class State(object):
    def __init__(self,
                 graph,
                 time_horizon,
                 max_time_horizon,
                 initial_locations):
        self._graph = graph
        self._time_horizon = time_horizon
        self._max_time_horizon = max_time_horizon
        self._attacker_location = initial_locations[0]
        self._defender_location = initial_locations[1]
        self.obs_attacker_location = copy.deepcopy(self._attacker_location)
        self._attacker_sequence = [copy.deepcopy(initial_locations[0])]
        self._defender_sequence = [copy.deepcopy(initial_locations[1])]
        self._defender_number = len(initial_locations[1])
        self.step = 0
        self.left_time = time_horizon

    def get_current_state(self):
        attacker_sequence = [self._attacker_location]
        state = copy.deepcopy(attacker_sequence)
        state += self._defender_location
        state.append(self.step)
        return state

    def get_observation(self):
        observation = []
        attacker_sequence = [self._attacker_location]
        state = copy.deepcopy(attacker_sequence)
        for player_id in range(self._defender_number):
            temp_observation_defender_location = [self._defender_location[player_id]]
            temp_observation = state + temp_observation_defender_location
            temp_observation.append(self.step)
            temp_observation.append(player_id)
            observation.append(copy.deepcopy(temp_observation))
        return observation

    def legal_actions(self):
        action_list = []
        for i in range(self._defender_number):
            action_list.append(self._graph.get_legal_action(self._defender_location[i]))
        return action_list

    def apply_action(self, attacker_path, defender_action):
        self._attacker_location = attacker_path[self.step + 1]
        self._attacker_sequence.append(copy.deepcopy(self._attacker_location))
        for i in range(len(self._defender_location)):
            self._defender_location[i] = self._graph.get_next_node(self._defender_location[i], defender_action[i])
        self._defender_sequence.append(copy.deepcopy(self._defender_location))
        self.left_time = self.left_time - 1
        self.step += 1

    def is_terminal(self):
        if self._attacker_location in self._defender_location or self._attacker_location in self._graph.exit_node:
            return True
        if self.left_time == 0:
            return True
        if self.left_time < 0:
            return "error"
        return False

    def rewards(self):
        if self.is_terminal():
            if self._attacker_location in self._defender_location:
                return [-1, 1]
            if self._attacker_location in self._graph.exit_node:
                return [1, -1]
            else:
                return [-1, 1]
        return [0, 0]
