# -*- coding: utf-8 -*-

import json
import numpy as np
import string
from .sarsa_lambda_agent import SarsaLambdaAgent
from lib_agent.lib_policy import *

class SarsaLambdaSMMwIMAgent(SarsaLambdaAgent):
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
    fl_beta : float
        the parameter in [0, 1] to control the effect of intrinsic motivation
    in_im_method : int
        the parameter to select IM function
    li_memory : tuple
        the memory as a sequence of events
    di_counts : dict
        a dictionary to keep count of events
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
        super(SarsaLambdaSMMwIMAgent, self).__init__(agent_id, learning_setting, policy, exp_no, configuration, observation_size, action_size, logger)

        self.in_regular_action_size = action_size
        self.li_memory_actions = self.cl_configuration.get_parameter("[memory]", "memory_actions").split()
        """The number of composite actions is the number of original actions times the number of memory actions"""
        self.in_action_size = self.in_regular_action_size * len(self.li_memory_actions)
        self.cl_policy = QPolicy(self.in_observation_size, self.in_action_size)  # Composite actions; Ex: (Left, Remember)

        self.in_memory_capacity = int(self.cl_configuration.get_parameter("[memory]", "memory_capacity"))
        self.st_event_content = self.cl_configuration.get_parameter("[memory]", "event_content")
        self.fl_beta = float(self.cl_configuration.get_parameter("[im]", "beta"))
        self.in_im_method = int(self.cl_configuration.get_parameter("[im]", "im_method"))
        self.in_im_start_episode = int(self.cl_configuration.get_possible_parameter("[im]", "im_start_episode", 0))

        self.li_memory = list()

        self.cl_experiment_logger.initialize_result_header("NumberOfMemoryChanges")
        self.cl_experiment_logger.initialize_result_header("SizeOfX")
        self.cl_experiment_logger.initialize_result_header("NumberOfVisitedX")
        self.cl_experiment_logger.initialize_result_header("TotalIntReward")

        self.li_episode_hist = []
        self.in_number_of_memory_changes = 0
        self.cl_visited_x = set()

        self.fl_total_int_reward = 0

        self.di_counts = {}
        self.di_x_counts = {}

        self.di_count_in_memory = {}

    def pre_episode(self):
        """
        Prepares the agent before the episode starts by
        initializing the memory and episodic history.
        """

        super().pre_episode()

        self.in_number_of_memory_changes = 0
        self.cl_visited_x = set()
        self.fl_total_int_reward = 0

        self.di_count_in_memory.clear()

        self.li_memory = list()
        if self._is_to_dump(self.in_episode_no):
            self.li_episode_hist = []
            self.li_episode_hist.append(('ea', 'ma', 'r', 'o', 'x', 'b'))

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

        """Increment the count of the event"""
        if self.tu_event_t not in self.di_counts.keys():
            self.di_counts[self.tu_event_t] = 0
        self.di_counts[self.tu_event_t] += 1

        """Form the state estimation at time t"""
        tu_estimated_state_t = (tuple(self.li_memory), tuple(observation))

        """Increment the count of the estimated state - action pair"""
        xa = (tu_estimated_state_t, action)
        if xa not in self.di_x_counts.keys():
            self.di_x_counts[xa] = 0
        self.di_x_counts[xa] += 1

        """Update the memory according to the memory action"""
        memory = tuple(self.li_memory)
        self.li_memory = self._update_memory(self.li_memory, self.tu_event_t, in_memory_action_index)
        memory_prime = tuple(self.li_memory)

        """Increment the number of memory changes if the memory is altered"""
        if memory != memory_prime:
            self.in_number_of_memory_changes += 1

        for i in range(0, len(self.li_memory)):
            tu_e = self.li_memory[i]
            if tu_e not in self.di_count_in_memory.keys():
                self.di_count_in_memory[tu_e] = 0
            self.di_count_in_memory[tu_e] += 1

        """Calculate intrinsic motivation and add it to the reward"""
        motivation = 0
        if self.in_episode_no >= self.in_im_start_episode:
            motivation = self._calculate_motivation(memory, tuple(observation), action, memory_prime, tuple(next_observation))
        fl_reward_with_motivation = reward + motivation

        self.fl_total_int_reward += motivation

        """Form the state estimate at time t+1 with the new memory"""
        tu_estimated_state_t_plus_1 = (tuple(self.li_memory), tuple(next_observation))

        """Set the estimated states at t and t+1 as visited"""
        self.cl_visited_x.add(tu_estimated_state_t)
        self.cl_visited_x.add(tu_estimated_state_t_plus_1)

        if self._is_to_dump(self.in_episode_no):
            table = str.maketrans('', '', string.ascii_lowercase)
            m_act = self.li_memory_actions[in_memory_action_index].translate(table)
            self.li_episode_hist.append((str(in_env_action), m_act, str(reward), str(tuple(next_observation)), str(tu_estimated_state_t_plus_1), str(motivation)))

        """Run regular Sarsa(lambda) updates on state estimations and reward with motivation"""
        super().post_action(tu_estimated_state_t, action, fl_reward_with_motivation, tu_estimated_state_t_plus_1, done)

    def post_episode(self):
        """
        Makes final changes on the agent side after the episode ends.
        Decays the epsilon if necessary.
        Dumps the episodic history and additional details about memory.
        Dumps the frequencies of the events.
        Logs the number of memory changes and the number of X in the Q-table.
        """

        if self._is_to_dump(self.in_episode_no):
            self._dump_frequencies()
            self._dump_history_info()

        self.cl_experiment_logger.set_result_log_value('NumberOfMemoryChanges', self.in_episode_no, self.in_number_of_memory_changes)
        self.cl_experiment_logger.set_result_log_value('NumberOfVisitedX', self.in_episode_no, len(self.cl_visited_x))
        self.cl_experiment_logger.set_result_log_value('SizeOfX', self.in_episode_no, len(self.cl_policy.get_q_table_keys()))
        self.cl_experiment_logger.set_result_log_value('TotalIntReward', self.in_episode_no, self.fl_total_int_reward)

        super().post_episode()

    def get_name(self):
        """
        Returns the name of the agent.

        Returns
        -------
        str
            the name of the agent formed with the lambda value, memory size, beta value and the agent id
        """
        table = str.maketrans('', '', string.ascii_lowercase)
        action_set = '_'.join([x.translate(table) for x in self.li_memory_actions])
        return "Sarsa({})-<{}>x{}-SMM-b{}-im{}-t{}-a{}-{}".format(self.fl_lambda, self.st_event_content, self.in_memory_capacity, self.fl_beta, self.in_im_method, self.in_im_start_episode, action_set, self.in_agent_id)

    def _update_memory(self, li_memory, tu_event, memory_action_index):
        """If the memory action is to push, add the event to the end of the memory"""
        if self.li_memory_actions[memory_action_index] == 'Push':
            if self.in_memory_capacity > 0:
                """If the memory is full, pop the event from the beginning"""
                if len(li_memory) >= self.in_memory_capacity:
                    li_memory = li_memory[1:]

                li_memory.append(tu_event)
            elif self.in_memory_capacity == -1: # ignore the capacity
                li_memory.append(tu_event)
        elif self.li_memory_actions[memory_action_index] == 'Clear+Push':
            if self.in_memory_capacity > 0 or self.in_memory_capacity == -1:
                """Clear the memory"""
                li_memory.clear()
                li_memory.append(tu_event)
        elif self.li_memory_actions[memory_action_index] == 'Clear':
            li_memory.clear()
        elif self.li_memory_actions[memory_action_index] == 'Pop':
            li_memory = li_memory[1:]
        elif self.li_memory_actions[memory_action_index] == 'Pop+Push':
            li_memory = li_memory[1:]
            if self.in_memory_capacity > 0:
                li_memory.append(tu_event)
        elif self.li_memory_actions[memory_action_index] == 'Add':
            if self.in_memory_capacity > 0:
                """If the memory is full, remove the event with lowest novelty"""
                if len(li_memory) >= self.in_memory_capacity:
                    min_ = 100 # dummy value
                    min_index = None
                    for i in range(0, len(li_memory)):
                        tu_e = li_memory[i]
                        v = pow(1.0 - self._get_probability(tu_e), self.di_count_in_memory[tu_e]) # decaying freq
                        if v < min_: # on ties, the oldest is selected
                            min_ = v
                            min_index = i

                    if min_index is not None:
                        del li_memory[min_index]

                li_memory.append(tu_event)
            elif self.in_memory_capacity == -1: # ignore the capacity
                li_memory.append(tu_event)
        elif self.li_memory_actions[memory_action_index] == 'Insert':
            if self.in_memory_capacity > 0:
                """If the memory is full, remove the event with highest freq"""
                if len(li_memory) >= self.in_memory_capacity:
                    max_ = -1 # dummy value
                    max_index = None
                    for i in range(0, len(li_memory)):
                        tu_e = li_memory[i]
                        v = self._get_probability(tu_e)
                        if v > max_: # on ties, the oldest is selected
                            max_ = v
                            max_index = i

                    if max_index is not None:
                        del li_memory[max_index]

                li_memory.append(tu_event)
            elif self.in_memory_capacity == -1: # ignore the capacity
                li_memory.append(tu_event)
        elif self.li_memory_actions[memory_action_index] == 'Enqueue':
            if self.in_memory_capacity > 0:
                """If the memory is full, remove the event with highest memory count"""
                if len(li_memory) >= self.in_memory_capacity:
                    max_ = -1 # dummy value
                    max_index = None
                    for i in range(0, len(li_memory)):
                        tu_e = li_memory[i]
                        v = self.di_count_in_memory[tu_e]
                        if v > max_: # on ties, the oldest is selected
                            max_ = v
                            max_index = i

                    if max_index is not None:
                        del li_memory[max_index]

                li_memory.append(tu_event)
            elif self.in_memory_capacity == -1: # ignore the capacity
                li_memory.append(tu_event)
        return li_memory

    def _dump_history_info(self):
        episode_debug_file = "{}/{}x{}.{}".format(self.st_debug_folder, self.in_experiment_no, self.in_episode_no, "rlmhist")
        with open(episode_debug_file, 'w') as f:
            for t in self.li_episode_hist:
                for v in t:
                    f.write("{}\t".format(v))
                f.write("\n")

    def _dump_frequencies(self):
        prob_debug_file = "{}/{}x{}.{}".format(self.st_debug_folder, self.in_experiment_no, self.in_episode_no, "rlfreq")
        di_counts = {}
        di_counts["event_counts"] = {}
        for k in sorted(self.di_counts.keys()):
            di_counts["event_counts"][str(k)] = self.di_counts[k]

        di_counts["event_counts_in_memory"] = {}
        for k in sorted(self.di_count_in_memory.keys()):
            di_counts["event_counts_in_memory"][str(k)] = self.di_count_in_memory[k]

        di_counts["x_counts"] = {}
        for k in sorted(self.di_x_counts.keys()):
            di_counts["x_counts"][str(k)] = self.di_x_counts[k]

        with open(prob_debug_file, 'w') as f:
            json.dump(di_counts, f, indent=4)

    def _get_probability(self, tu_e):
        if tu_e in self.di_counts.keys():
            return self.di_counts[tu_e] / sum(self.di_counts.values())
        return 0

    def _calculate_frequency_motivation(self, memory, memory_prime):
        bonus = (-1) * self.in_memory_capacity
        for i in range(0, len(memory_prime)):
            tu_e = memory_prime[i]
            bonus += (1 - self._get_probability(tu_e))
        return bonus

    def _calculate_decaying_frequency_motivation(self, memory, memory_prime):
        m_sum = 0
        for i in range(0, len(memory_prime)):
            tu_e = memory_prime[i]
            m_sum += pow(1.0 - self._get_probability(tu_e), self.di_count_in_memory[tu_e])
        m_val = (m_sum / len(memory_prime)) if len(memory_prime) > 0 else 0
        return m_val - 1

    def _calculate_count_based_motivation(self, x, a):
        xa = (x, a)
        return 1.0 / np.sqrt(self.di_x_counts[xa])

    def _calculate_motivation(self, memory, o, a, memory_prime, o_p):
        if self.in_im_method == 1:
            return self.fl_beta * self._calculate_frequency_motivation(memory, memory_prime)
        elif self.in_im_method == 2:
            return self.fl_beta * self._calculate_decaying_frequency_motivation(memory, memory_prime)
        elif self.in_im_method == 3:
            x = (tuple(memory), o)
            return self.fl_beta * self._calculate_count_based_motivation(x, a)
        return 0
