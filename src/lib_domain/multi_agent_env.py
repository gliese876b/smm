import gym
import numpy as np


class MultiAgentEnv(gym.Env):
    """ MultiAgent version for a gym.Env class"""

    def reset(self):
        """ resets the env and returns the initial observations for the agents

            returns --> obs_dict: dict that maps each agent_id string to an observation
        """
        raise NotImplementedError

    def step(self, action_dict):
        """ one step simulation of each agent

            param1 --> action_dict : dict that maps each agent_id string to an action

            returns -->  obs_dict : dict that maps each agent_id string to an observation
                         rew_dict : dict that maps each agent_id string to a reward
                         done_dict: dict that maps each agent_id string to a bool value
                                    indicating the agent is done or not. "__all__" is required
                                    (as in rllib) to check env termination.
                         info_dict: optional information dict
        """
        raise NotImplementedError

    def get_observation_space_size(self):
        """
            returns --> observation_space_size: int
        """
        raise NotImplementedError

    def get_action_space_size(self):
        """
            returns --> action_space_size: int
        """
        raise NotImplementedError

    def set_agent_names(self, names):
        self.li_agent_names = names

    def set_logger(self, logger):
        self.cl_std_logger = logger

    def get_states_dict(self):
        states_dict = {}
        for agent_name in self.li_agent_names:
            states_dict[agent_name] = np.asarray(self.di_current_states[agent_name], dtype=object)
        return states_dict

    def get_observation_labels(self):
        return {}
