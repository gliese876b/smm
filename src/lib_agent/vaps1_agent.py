# -*- coding: utf-8 -*-
import random
import numpy as np
from functools import cmp_to_key
from .abstract_agent import AbstractAgent
import json
from .lib_policy import QPolicy


class VAPS1Agent(AbstractAgent):
    """
    The implementation of VAPS(1) algorithm proposed in
    Peshkin, L., Meuleau, N., & Kaelbling, L. (1999). Learning Policies with External Memory. Retrieved from http://arxiv.org/abs/cs/0103003
    This implementation follows the composite action space semantic.

    Attributes
    ----------
    in_memory_size : int
        the number of bits in the memory
    fl_boltzmann_temperature : float
        the initial temperature value
    fl_temperature_end : float
        the final temperature value
    fl_b : float
        b value from the algorithm
    fl_temperature_decay : float
        the temperature decay value to be used at the end of each episode
    in_regular_action_size : int
        the number of actions that can be taken in the environment
    in_action_size : int
        the number of composite actions
    cl_q_values : QPolicy
        the policy that the agent updates and follows

    """
    def __init__(self, agent_id, learning_setting, policy, exp_no, configuration, observation_size, action_size, logger):
        """
        Parameters
        ----------
        agent_id : int
            the ID number of the agent
        learning_setting : str
            learning related settings
        policy : AbstractPolicy
            the initial policy to be followed (Default value = None)
        exp_no : int
            the experiment number
        configuration : dict
            the configuration of the experiment with the necessary parameters
        observation_size : int
            the size of the observation vector
        action_size : int
            the number of actions
        logger: ExperimentLogger
            the experiment logger to keep logged data during the experiment
        """

        super(VAPS1Agent, self).__init__(agent_id, learning_setting, exp_no, configuration, observation_size, action_size, logger)

        self.in_memory_size = int(self.cl_configuration.get_parameter("[vaps]", "memory_size"))
        self.fl_boltzmann_temperature = float(self.cl_configuration.get_parameter("[vaps]", "boltzmann_temperature_start"))
        self.fl_temperature_end = float(self.cl_configuration.get_parameter("[vaps]", "boltzmann_temperature_end"))
        self.fl_b = float(self.cl_configuration.get_parameter("[vaps]", "b"))
        self.fl_temperature_decay = (self.fl_boltzmann_temperature - self.fl_temperature_end) / (self.in_number_of_episodes-1) if self.in_number_of_episodes > 1 else 0
        
        self.in_regular_action_size = action_size
        """The number of composite actions is the number of original actions times 2^n where n is the number of bits in memory"""
        self.in_action_size = self.in_regular_action_size * pow(2, self.in_memory_size)
        self.cl_q_values = QPolicy(0, self.in_action_size)
        
        self.cl_experiment_logger.initialize_result_header("NumberOfMemoryChanges")
        self.cl_experiment_logger.initialize_result_header("BoltzmannTemperature")
        self.cl_experiment_logger.initialize_result_header("SizeOfX")

        self.in_number_of_memory_changes = 0
        self.tu_memory = tuple([0] * self.in_memory_size)
        self.di_trace = {}
        self.in_time = 0
        self.li_episode_hist = []

    def pre_episode(self):
        """
        Prepares the agent before the episode starts by
        initializing the memory and episodic history.
        """

        super().pre_episode()
        
        """Initialize the memory as a binary string"""
        self.in_number_of_memory_changes = 0
        self.tu_memory = tuple([0] * self.in_memory_size)
        self.di_trace = {}
        self.in_time = 0

        if self._is_to_dump(self.in_episode_no):
            self.li_episode_hist = []
            self.li_episode_hist.append(('ea', 'ma', 'r', 'o', 'm'))        
        
    def pre_action(self, observation):
        """
        Returns the action selected by the agent on the
        augmented observation formed by concatenating the binary memory
        and the current observation

        Parameters
        ----------
        observation : numpy array
            the current observation

        Returns
        -------
        int
            selected action for the given observation
        """
        ao = tuple(self._augment_observation(observation, self.tu_memory))
        
        self.cl_q_values.init_q_values(ao)
        self._init_trace(ao)

        """Select action according to boltzmann law"""
        self.in_selected_action = np.random.choice(self.in_action_size, 1, p=self._to_prob_distribution(self.cl_q_values.get_q_values(ao)))[0]    # select action according to the augmented observation
        
        env_action, _ = self._separate_actions(self.in_selected_action)
        
        if self._is_to_dump(self.in_episode_no) and len(self.li_episode_hist) == 1:
            self.li_episode_hist.append(('-', '-', '-', str(tuple(observation)), str(self.tu_memory)))            
        """The agent chooses a composite action but only the environmental action is returned"""
        return env_action
        
    def post_action(self, observation, action, reward, next_observation, done):
        """
        Makes VAPS(1) updates on observations augmented with memory.
        The memory is also updated according to the memory action.

        Parameters
        ----------
        observation : numpy array
            the observation at time t
        action : int
            the action taken at time t
        reward : float
            the reward taken at time t+1
        next_observation : numpy array
            the observation at time t+1
        done : bool
            bool determining if the episode is ended
        """

        """Augment the observation at time t with the memory at time t"""
        ao = tuple(self._augment_observation(observation, self.tu_memory))

        """Calculate the probability distribution on augmented observation"""
        ao_pd = self._to_prob_distribution(self.cl_q_values.get_q_values(ao))
        """Update the exploration trace"""
        for a in range(self.in_action_size):
            delta_trace = 0
            if a == self.in_selected_action:
                delta_trace = (1.0 - ao_pd[a]) / self.fl_boltzmann_temperature
            else:
                delta_trace = - ao_pd[a] / self.fl_boltzmann_temperature
            
            self.di_trace[ao][a] += delta_trace

        """Calculate instantaneous error"""
        fl_error = self.fl_b - np.power(self.fl_discount_rate, self.in_time) * reward
        """Make updates on Q-values"""
        for ao_, q_values in self.cl_q_values.get_q_table_items():
            for a in range(self.in_action_size):
                delta_q = - self.fl_learning_rate * fl_error * self._get_trace(ao_, a)
                q_value = q_values[a]
                self.cl_q_values.set_q_value(ao_, a, q_value + delta_q)

        self.in_time += 1
        
        """Make memory update according to the memory action"""
        memory = self.tu_memory
        env_action, mem_action = self._separate_actions(self.in_selected_action)
        binary_list = list(map(int, list(np.binary_repr(mem_action))))
        memory_prime = tuple([0] * (self.in_memory_size - len(binary_list)) + binary_list)
        self.tu_memory = memory_prime

        """Increment the number of memory changes if necessary"""
        if memory != memory_prime:
            self.in_number_of_memory_changes += 1        
        
        if self._is_to_dump(self.in_episode_no):
            self.li_episode_hist.append((str(env_action), str(memory_prime), str(reward), str(tuple(next_observation)), str(self.tu_memory)))          

        super().post_action(observation, action, reward, next_observation, done)
    
    def post_episode(self):
        """
        Makes final changes on the agent side after the episode ends.
        Decays the boltzmann temperature.
        Dumps the episodic history.
        Dumps the policy.
        Logs the number of memory changes and the number of X in the Q-table.
        """

        if self._is_to_dump(self.in_episode_no):
            self._dump_history_info()
            self._dump_policy()

        self.cl_experiment_logger.set_result_log_value('NumberOfMemoryChanges', self.in_episode_no, self.in_number_of_memory_changes)
        self.cl_experiment_logger.set_result_log_value('BoltzmannTemperature', self.in_episode_no, self.fl_boltzmann_temperature)
        self.cl_experiment_logger.set_result_log_value('SizeOfX', self.in_episode_no, len(self.cl_q_values.get_q_table_keys()))

        if (self.fl_boltzmann_temperature - self.fl_temperature_decay) > 0.00001:
            self.fl_boltzmann_temperature -= self.fl_temperature_decay
        else:
            self.fl_boltzmann_temperature = 0
            
        super().post_episode()
        
    def get_name(self):
        """
        Returns the name of the agent.

        Returns
        -------
        str
            the name of the agent formed with the memory size and the agent id
        """
        return "VAPS(1)-{}-bit-{}".format(self.in_memory_size, self.in_agent_id)

    def _init_trace(self, augmented_observation):
        obs_key = tuple(augmented_observation)
        if obs_key not in self.di_trace:
            self.di_trace[obs_key] = np.zeros((self.in_action_size,))

    def _get_trace(self, augmented_observation, a):
        obs_key = tuple(augmented_observation)
        if obs_key not in self.di_trace.keys():
            self.di_trace[obs_key] = np.zeros((self.in_action_size,))
        return self.di_trace[obs_key][a]
    
    def _augment_observation(self, observation, memory):
        return np.concatenate([np.asarray(observation), np.asarray(memory)])
        
    def _separate_actions(self, selected_action):
        number_of_memory_combinations = pow(2, self.in_memory_size)
        env_action = int(selected_action / number_of_memory_combinations)
        mem_action = selected_action % number_of_memory_combinations
        return env_action, mem_action

    def _to_prob_distribution(self, l):
        array = np.exp(np.asarray(l) / self.fl_boltzmann_temperature)
        assert(np.sum(array) > 0)
        return array / np.sum(array)
        
    def _dump_history_info(self):
        episode_debug_file = "{}/{}x{}.{}".format(self.st_debug_folder, self.in_experiment_no, self.in_episode_no, "rlmhist")
        with open(episode_debug_file, 'w') as f:
            for t in self.li_episode_hist:
                for v in t:
                    f.write("{}\t".format(v))
                f.write("\n")
    
    def _dump_policy(self):
        episode_debug_file = "{}/{}x{}.{}".format(self.st_debug_folder, self.in_experiment_no, self.in_episode_no, "rlpol")
        di_to_dump = {}
        di_to_dump["policy"] = {}
        di_to_dump["q_table"] = {}
        di_to_dump["trace"] = {}
        for ao in sorted(self.cl_q_values.get_q_table_keys(),key=cmp_to_key(QPolicy._sort_q_table_keys)):
            di_to_dump["policy"][str(ao)] = {}
            di_to_dump["q_table"][str(ao)] = {}
            di_to_dump["trace"][str(ao)] = {}
            p = self._to_prob_distribution(self.cl_q_values.get_q_values(ao))
            for a in range(self.in_action_size):
                ea, ma = self._separate_actions(a)
                binary_list = list(map(int, list(np.binary_repr(ma))))
                ma = tuple([0] * (self.in_memory_size - len(binary_list)) + binary_list)
                tu_a = (ea, ma)
                di_to_dump["q_table"][str(ao)][str(tu_a)] = str(self.cl_q_values.get_q_values(ao)[a])
                di_to_dump["policy"][str(ao)][str(tu_a)] = p[a]
                di_to_dump["trace"][str(ao)][str(tu_a)] = self._get_trace(ao, a)

        with open(episode_debug_file, 'w') as f:
            json.dump(di_to_dump, f, indent=4)
