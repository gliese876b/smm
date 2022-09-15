from abc import ABC, abstractmethod


class AbstractPolicy(ABC):
    @abstractmethod
    def __init__(self, observation_size, action_size):
        self.in_observation_size = observation_size
        self.in_action_size = action_size

    # every policy should implement the selection mechanism
    @abstractmethod
    def get_action(self, observation):
        raise NotImplementedError

    def dump_policy(self, debug_folder, experiment_no, episode_no):
        pass

    def read_policy(self, file_path):
        pass

    def set_policy_name(self, name):
        self.st_name = name
