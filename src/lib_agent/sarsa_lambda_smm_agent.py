# -*- coding: utf-8 -*-

import json
import numpy as np
from .sarsa_lambda_agent import SarsaLambdaAgent
from lib_agent.lib_policy import *

class SarsaLambdaSMMAgent(SarsaLambdaAgent):
    """
    Sarsa lambda agent with Self Memory Management where there is a joint policy with composite actions.
    The set of domain actions is composed with the set of memory actions.

    Attributes
    ----------
    in_regular_action_size : int
        the number of actions that can be taken in the environment
    li_memory_actions : list
        the list of memory actions that the agent can take
    in_action_size : int
        the number of composite actions
    cl_policy : QPolicy
        the policy that the agent updates and follows
    in_memory_capacity : int
        the size of the memory
    event_content : str
        the content of an event to be included in the memory
    li_memory : tuple
        the memory as a sequence of events
    li_episode_hist : list
        a sequence of episodic transitions including memory
    in_number_of_memory_changes : int
        the number of changes on memory
    cl_visited_x : set
        a set of visited state estimations (used for faster Q-table update)

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
        super(SarsaLambdaSMMAgent, self).__init__(agent_id, learning_setting, policy, exp_no, configuration, observation_size, action_size, logger)

        self.in_regular_action_size = action_size
        self.li_memory_actions = self.cl_configuration.get_parameter("[memory]", "memory_actions").split()
        """The number of composite actions is the number of original actions times the number of memory actions"""
        self.in_action_size = self.in_regular_action_size * len(self.li_memory_actions)
        self.cl_policy = QPolicy(self.in_observation_size, self.in_action_size)  # Composite actions; Ex: (Left, Remember)

        self.in_memory_capacity = int(self.cl_configuration.get_parameter("[memory]", "memory_capacity"))
        self.st_event_content = self.cl_configuration.get_parameter("[memory]", "event_content")

        self.li_memory = list()
        
        self.cl_experiment_logger.initialize_result_header("NumberOfMemoryChanges")
        self.cl_experiment_logger.initialize_result_header("SizeOfX")
        self.cl_experiment_logger.initialize_result_header("NumberOfVisitedX")
        
        self.li_episode_hist = []
        self.in_number_of_memory_changes = 0
        self.cl_visited_x = set()

    def pre_episode(self):
        """
        Prepares the agent before the episode starts by
        initializing the memory and episodic history.
        """

        super().pre_episode()

        self.in_number_of_memory_changes = 0
        self.cl_visited_x = set()
        self.li_memory = list()
        if self._is_to_dump(self.in_episode_no):
            self.li_episode_hist = []
            self.li_episode_hist.append(('ea', 'ma', 'r', 'o', 'x'))
        
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
        tu_estimated_state_t = (tuple(self.li_memory), tuple(observation))

        if self._is_to_dump(self.in_episode_no) and len(self.li_episode_hist) == 1:
            self.li_episode_hist.append(('-', '-', '-', '-', str(tuple(observation)), str(tu_estimated_state_t)))
    
        in_composite_action = super().pre_action(tu_estimated_state_t)
        """The agent chooses a composite action but only the environmental action is returned"""
        return in_composite_action % self.in_regular_action_size
        
    def post_action(self, observation, action, reward, next_observation, done):
        """
        Makes Sarsa(lambda) updates on Q and eligibility trace values
        based on the experienced transition where the transition occurs between
        state estimations as (x_t, a_t, r_{t+1}, x_{t+1}). The memory is also updated
        according to the memory action.
        Updates the counts on the observation.

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

        """Separate the environmental and memory actions"""
        in_env_action = self.in_action % self.in_regular_action_size
        in_memory_action_index = self.in_action // self.in_regular_action_size

        """Form the event of time t"""
        self.tu_event_t = None
        if 'a' in self.st_event_content:
            self.tu_event_t = tuple([tuple(observation), in_env_action])
        else:
            self.tu_event_t = tuple(observation)

        """Form the state estimation at time t"""
        tu_estimated_state_t = (tuple(self.li_memory), tuple(observation))

        """Update the memory according to the memory action"""
        memory = tuple(self.li_memory)
        self.li_memory = self._update_memory(self.li_memory, self.tu_event_t, in_memory_action_index)
        memory_prime = tuple(self.li_memory)

        """Increment the number of memory changes if the memory is altered"""
        if memory != memory_prime:
            self.in_number_of_memory_changes += 1

        """Form the state estimate at time t+1 with the new memory"""
        tu_estimated_state_t_plus_1 = (tuple(self.li_memory), tuple(next_observation))

        """Set the estimated states at t and t+1 as visited"""
        self.cl_visited_x.add(tu_estimated_state_t)
        self.cl_visited_x.add(tu_estimated_state_t_plus_1)

        if self._is_to_dump(self.in_episode_no):
            self.li_episode_hist.append((str(in_env_action), self.li_memory_actions[in_memory_action_index][0], str(reward), str(tuple(next_observation)), str(tu_estimated_state_t_plus_1)))

        """Run regular Sarsa(lambda) updates on state estimations and reward with motivation"""
        super().post_action(tu_estimated_state_t, action, reward, tu_estimated_state_t_plus_1, done)
        
    def post_episode(self):
        """
        Makes final changes on the agent side after the episode ends.
        Decays the epsilon if necessary.
        Dumps the episodic history and additional details about memory.
        Dumps the frequencies of the events.
        Logs the number of memory changes and the number of X in the Q-table.
        """

        if self._is_to_dump(self.in_episode_no):
            self._dump_history_info()
    
        self.cl_experiment_logger.set_result_log_value('NumberOfMemoryChanges', self.in_episode_no, self.in_number_of_memory_changes)
        self.cl_experiment_logger.set_result_log_value('NumberOfVisitedX', self.in_episode_no, len(self.cl_visited_x))
        self.cl_experiment_logger.set_result_log_value('SizeOfX', self.in_episode_no, len(self.cl_policy.get_q_table_keys()))
        super().post_episode()  
        
    def get_name(self):
        """
        Returns the name of the agent.

        Returns
        -------
        str
            the name of the agent formed with the lambda value, memory size, beta value and the agent id
        """
        return "Sarsa({})-<{}>x{}-SMM-{}".format(self.fl_lambda, self.st_event_content, self.in_memory_capacity, self.in_agent_id)

    def _dump_history_info(self):
        episode_debug_file = "{}/{}x{}.{}".format(self.st_debug_folder, self.in_experiment_no, self.in_episode_no, "rlmhist")
        with open(episode_debug_file, 'w') as f:
            for t in self.li_episode_hist:
                for v in t:
                    f.write("{}\t".format(v))
                f.write("\n")

    def _update_memory(self, li_memory, tu_event, memory_action_index):
        """If the memory action is to push, add the event to the end of the memory"""
        if self.li_memory_actions[memory_action_index] == 'Push':
            if self.in_memory_capacity > 0:
                """If the memory is full, pop the event from the beginning"""
                if len(li_memory) >= self.in_memory_capacity:
                    li_memory = li_memory[1:]

                li_memory.append(tu_event)
        return li_memory
