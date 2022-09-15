# -*- coding: utf-8 -*-
import random
import numpy as np
import math
from .sarsa_lambda_agent import SarsaLambdaAgent
from lib_agent.lib_policy import *
from lib_agent.lib_mc import *


class SarsaLambdaMemoryBasedAgent(SarsaLambdaAgent):
    """
    A class for an agent following Sarsa(lambda) algorithm with an external memory.

    Attributes
    ----------
    st_mc_type : str
        type of the memory controller
    in_memory_capacity : int
        size of the memory
    cl_memory_controller : AbstractMemoryController
        an instance of a memory controller class
    li_episode_hist : list
        the sequence of transitions done in an episode

    Methods
    -------
    dump_history_info()
        dumps the episodic history including the content of the memory
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
        super(SarsaLambdaMemoryBasedAgent, self).__init__(agent_id, learning_setting, policy, exp_no, configuration, observation_size, action_size, logger)

        self.cl_experiment_logger.initialize_result_header("SizeOfX")
        self.cl_experiment_logger.initialize_result_header("NumberOfMemoryChanges")
        
        self.st_mc_type = self.cl_configuration.get_parameter("[memory]", "memory_controller_type")
        self.in_memory_capacity = int(self.cl_configuration.get_parameter("[memory]", "memory_capacity"))
        self.st_event_content = self.cl_configuration.get_parameter("[memory]", "event_content")
        
        self.cl_memory_controller = self._get_memory_controller()
        self.li_episode_hist = []

    def pre_episode(self):
        """
        Prepares the agent before the episode starts by
        initializing the memory and episodic history.
        """

        super().pre_episode()
        self.cl_memory_controller.clear_memory()
        
        if self._is_to_dump(self.in_episode_no):
            self.li_episode_hist = []
            self.li_episode_hist.append(('a', 'r', 'o', 'x'))
        
    def pre_action(self, observation):
        """
        Returns the action selected by the agent on the
        state estimation formed by concatenating the memory
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

        tu_estimated_state = self.cl_memory_controller.get_state_estimation(observation)
        
        if self._is_to_dump(self.in_episode_no) and len(self.li_episode_hist) == 1:
            self.li_episode_hist.append(('-', '-', str(tuple(observation)), str(tu_estimated_state), '-'))        
    
        return super().pre_action(tu_estimated_state)
        
    def post_action(self, observation, action, reward, next_observation, done):
        """
        Makes Sarsa(lambda) updates on Q and eligibility trace values
        based on the experienced transition where the transition occurs between
        state estimations as (x_t, a_t, r_{t+1}, x_{t+1}). The memory is also updated
        according to the memory controller.

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

        tu_estimated_state_t = self.cl_memory_controller.get_state_estimation(observation)
        
        self.cl_memory_controller.update_memory(observation, action, reward, next_observation, done)

        tu_estimated_state_t_plus_1 = self.cl_memory_controller.get_state_estimation(next_observation)

        if self._is_to_dump(self.in_episode_no):
            self.li_episode_hist.append((str(action), str(reward), str(tuple(next_observation)), str(tu_estimated_state_t_plus_1)))
        
        super().post_action(tu_estimated_state_t, action, reward, tu_estimated_state_t_plus_1, done)
        
    def post_episode(self):
        """
        Makes final changes on the agent side after the episode ends.
        Decays the epsilon if necessary.
        Dumps the episodic history and additional details about memory.
        Logs the number of memory changes and the number of X in the Q-table.
        """

        if self._is_to_dump(self.in_episode_no):
            self.dump_history_info()
            self.cl_memory_controller.dump_details(self.st_debug_folder, self.in_experiment_no, self.in_episode_no)
    
        self.cl_experiment_logger.set_result_log_value('SizeOfX', self.in_episode_no, len(self.cl_policy.get_q_table_keys()))
        self.cl_experiment_logger.set_result_log_value('NumberOfMemoryChanges', self.in_episode_no, self.cl_memory_controller.get_number_of_memory_updates())
        super().post_episode()  
        
    def get_name(self):
        """
        Returns the name of the agent.

        Returns
        -------
        str
            the name of the agent formed with the lambda value, memory size, controller type and the agent id
        """
        return "Sarsa({})-<{}>x{}-MB-{}-{}".format(self.fl_lambda, self.st_event_content, self.in_memory_capacity, self.st_mc_type, self.in_agent_id)

    def dump_history_info(self):
        """
        Dumps the episodic history into a file with rlmhist extension
        """

        episode_debug_file = "{}/{}x{}.{}".format(self.st_debug_folder, self.in_experiment_no, self.in_episode_no, "rlmhist")
        with open(episode_debug_file, 'w') as f:
            for t in self.li_episode_hist:
                for v in t:
                    f.write("{}\t".format(v))
                f.write("\n")
                
    def _get_memory_controller(self):
        if self.st_mc_type == "fixed_window":
            return MemoryControllerFixedWindow(self.cl_configuration, self.cl_experiment_logger)
        if self.st_mc_type == "random_fixed_window":
            return MemoryControllerRandomFixedWindow(self.cl_configuration, self.cl_experiment_logger)    
        return None
