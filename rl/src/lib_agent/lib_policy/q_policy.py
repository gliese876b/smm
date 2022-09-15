from .abstract_policy import AbstractPolicy
from functools import cmp_to_key
import numpy as np
import json


class QPolicy(AbstractPolicy):
    def __init__(self, observation_size, action_size):
        super(QPolicy, self).__init__(observation_size, action_size)

        self.di_q_table = {}

    def init_q_values(self, observation):
        obs_key = tuple(observation)  # convert the array of (1, observation_size) to a hashable object (tuple)
        if obs_key not in self.di_q_table:
            self.di_q_table[obs_key] = np.zeros((self.in_action_size,))

    def set_q_values(self, observation, values):
        obs_key = tuple(observation)  # convert the array of (1, observation_size) to a hashable object (tuple)
        if obs_key not in self.di_q_table:
            self.di_q_table[obs_key] = values            

    def get_q_values(self, observation):
        obs_key = tuple(observation)  # convert the array of (1, observation_size) to a hashable object (tuple)
        if obs_key not in self.di_q_table:
            self.di_q_table[obs_key] = np.zeros((self.in_action_size,))
        return self.di_q_table[obs_key]

    def get_q_table_keys(self):
        return self.di_q_table.keys()

    def get_q_table_items(self):
        return self.di_q_table.items()

    def get_max_q_value(self, observation):
        return np.max(self.get_q_values(observation))

    def get_q_value(self, observation, action):
        obs_key = tuple(observation)  # convert the array of (1, observation_size) to a hashable object (tuple)
        return self.di_q_table[obs_key][action]

    def set_q_value(self, observation, action, value):
        obs_key = tuple(observation)  # convert the array of (1, observation_size) to a hashable object (tuple)
        self.di_q_table[obs_key][action] = value

    def get_action(self, observation):
        return np.random.choice(np.flatnonzero(self.get_q_values(observation) == self.get_q_values(observation).max()))

    @staticmethod
    def _sort_q_table_keys(a, b):
        if len(a) > len(b):
            return 1
        elif len(a) == len(b):
            i = 0
            while i < len(a):
                if type(a[i]) != type(b[i]):    # if a tuple and int needed to be compared in the q_table
                    if type(a[i]) == int:
                        return -1
                    else:
                        return 1
                else:
                    if a[i] > b[i]:
                        return 1
                    elif a[i] == b[i]:
                        i += 1
                        continue
                    else:
                        return -1
            return 0        # the code is not supposed to reach this point, just to be safe.
        else:
            return -1

    def dump_policy(self, debug_folder, experiment_no, episode_no, di_to_dump={}):
        episode_debug_file = "{}/{}x{}.{}".format(debug_folder, experiment_no, episode_no, "rlpol")
        di_to_dump["policy"] = {}
        di_to_dump["q_table"] = {}
        for o in sorted(self.di_q_table.keys(),key=cmp_to_key(self._sort_q_table_keys)):
            di_to_dump["policy"][str(o)] = str(np.argmax(self.di_q_table[o]))
            di_to_dump["q_table"][str(o)] = {}
            for a in range(self.in_action_size):
                di_to_dump["q_table"][str(o)][str(a)] = str(self.di_q_table[o][a])

        with open(episode_debug_file, 'w') as f:
            json.dump(di_to_dump, f, indent=4)

    def read_policy(self, file_path):
        with open(file_path, 'r') as json_file:
            di_to_load = json.load(json_file)
            for st_obs in di_to_load["q_table"].keys():
                tu_obs = eval(st_obs)   # casting to tuple of integers
                self.di_q_table[tu_obs] = np.zeros((self.in_action_size,))
                for a in range(self.in_action_size):
                    self.di_q_table[tu_obs][a] = float(di_to_load["q_table"][st_obs][str(a)])