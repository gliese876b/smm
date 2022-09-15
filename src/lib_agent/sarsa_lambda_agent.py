# -*- coding: utf-8 -*-
import random
import numpy as np
from .abstract_agent import AbstractAgent
from .lib_policy import QPolicy
from functools import cmp_to_key


class SarsaLambdaAgent(AbstractAgent):
    """
    A class for an agent following Sarsa(lambda) algorithm.

    Attributes
    ----------
    cl_policy : AbstractPolicy
        the policy that the agent follows
    di_e_table : dict
        a dictionary to keep eligibility traces
    fl_lambda : float
        a float for trace decay parameter
    in_action : int
        the action selected for the current time step
    in_next_action : int
        the action selected for the next time step
    bo_random_action : bool
        boolean representing if the current action is selected randomly
    cl_visited_pairs : set
        the set of visited observation/estimated state and action pairs
        to be used in the update phase of the pairs with positive traces

    Methods
    -------
    pre_episode()
        prepares the agent before the episode starts
    pre_action(observation)
        returns the action selected by the agent on the given observation
    post_action(observation, action, reward, next_observation, done)
        makes Sarsa(lambda) updates on Q and eligibility trace values
        based on the experienced transition
    post_episode()
        makes final changes on the agent side after the episode ends
    get_name()
        returns the name of the agent
    init_e_values(observation)
        initializes eligibility trace values to zero for the given observation
    get_e_values(observation)
        returns an array of eligibility traces for each action
    get_e_value(observation, action)
        returns the eligibility trace for the given observation-action pair
    set_e_value(observation, action, value)
        sets the given eligibility trace value for the given observation-action pair
    reset_e_values()
        resets all eligibility traces to zero
    select_action(observation)
        employs and epsilon-greedy action selection and returns the selected action
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

        super(SarsaLambdaAgent, self).__init__(agent_id, learning_setting, exp_no, configuration, observation_size, action_size, logger)
        self.cl_policy = policy
        self.di_e_table = {}
        self.fl_lambda = float(self.cl_configuration.get_parameter(learning_setting, "lambda"))

        # dummy initializations
        self.in_action = -1
        self.in_next_action = -1
        self.bo_random_action = True
        
        self.cl_visited_pairs = set()

    def pre_episode(self):
        """
        Prepares the agent before the episode starts by
        initializing the current and next actions and resetting
        the eligibility traces.
        """
        super().pre_episode()
        
        self.in_action = -1
        self.in_next_action = -1
        
        self.reset_e_values()
        
    def pre_action(self, x):
        """
        Returns the action selected by the agent on the given x.

        Parameters
        ----------
        x : numpy array
            the current x

        Returns
        -------
        int
            selected action for the given x
        """

        if self.in_action == -1:
            self.in_action = self.select_action(tuple(x))  # initial action
        return self.in_action

    def post_action(self, x_t, a_t, r_t, x_t_plus_1, done):
        """
        Makes Sarsa(lambda) updates on Q and eligibility trace values
        based on the experienced transition. The update is only employed on
        the set of visited pairs for optimization since those would be the only
        ones with positive eligibility traces.

        Parameters
        ----------
        x_t : numpy array
            the x at time t
        a_t : int
            the action taken at time t
        r_t : float
            the reward taken at time t+1
        x_t_plus_1 : numpy array
            the observation at time t+1
        done : bool
            bool determining if the episode is ended
        """
        self.in_next_action = self.select_action(tuple(x_t_plus_1))
        
        delta = r_t + self.fl_discount_rate * self.cl_policy.get_q_value(tuple(x_t_plus_1), self.in_next_action) \
            - self.cl_policy.get_q_value(tuple(x_t), self.in_action)

        self.set_e_value(tuple(x_t), self.in_action, 1.0)
        self.cl_visited_pairs.add((tuple(x_t), self.in_action))
        for x_, a_ in self.cl_visited_pairs:
            e_value = self.di_e_table[x_][a_]
            new_e_value = e_value
            if not (tuple(x_) == tuple(x_t) and a_ == self.in_action):
                new_e_value = self.fl_discount_rate * self.fl_lambda * e_value
            self.set_e_value(x_, a_, new_e_value)    
            q_v = self.cl_policy.get_q_value(x_, a_)
            q_update = self.fl_learning_rate * delta * new_e_value
            self.cl_policy.set_q_value(x_, a_, q_v + q_update)
        self.in_action = self.in_next_action  # set the action to be executed
        super().post_action(tuple(x_t), a_t, r_t, tuple(x_t_plus_1), done)
        
    def post_episode(self):
        """
        Makes final changes on the agent side after the episode ends.
        Decays the epsilon if necessary.
        Dumps the policy to a file if it is set to be dumped.
        """
        if (self.fl_epsilon - self.fl_epsilon_decay) > 0.00001:
            self.fl_epsilon -= self.fl_epsilon_decay
        else:
            self.fl_epsilon = 0
        
        if self._is_to_dump(self.in_episode_no):
            self._dump_policy()
        
        super().post_episode()  
        
    def get_name(self):
        """
        Returns the name of the agent.

        Returns
        -------
        str
            the name of the agent formed with the lambda value and the agent id
        """
        return "Sarsa({})-{}".format(self.fl_lambda, self.in_agent_id)

    def init_e_values(self, x):
        """
        Initializes eligibility trace values to zero for the given x.

        Parameters
        ----------
        x : numpy array
            the x whose eligibility traces are to be initialized
        """

        if x not in self.di_e_table:
            self.di_e_table[x] = np.zeros((self.in_action_size,))
        
    def get_e_values(self, x):
        """
        Returns an array of eligibility traces for each action.

        Parameters
        ----------
        x : numpy array
            the x whose eligibility traces are to be get

        Returns
        -------
        numpy array
            an array that holds eligibility trace of an action
            in the position indexed by the action
        """

        if x not in self.di_e_table:
            self.di_e_table[x] = np.zeros((self.in_action_size,))
        return self.di_e_table[x]

    def get_e_value(self, x, a):
        """
        Returns the eligibility trace for the given x-a pair.

        Parameters
        ----------
        x : numpy array
            the x whose eligibility traces are to be get
        a : int
            the action to get the eligibility trace value

        Returns
        -------
        float
            the eligibility trace value for the x-a pair
        """

        if x not in self.di_e_table.keys():   # observation coming from another agent
            return 0
        return self.di_e_table[x][a]

    def set_e_value(self, x, a, value):
        """
        Sets the given eligibility trace value for the given x-a pair.

        Parameters
        ----------
        x : numpy array
            the observation whose eligibility traces are to be set
        a : int
            the action to set the eligibility trace value
        value : float
            the eligibility trace value to be set for
            the given observation-action pair

        """
        if x not in self.di_e_table.keys():   # observation coming from another agent
            return
        self.di_e_table[x][a] = value

    def reset_e_values(self):
        """
        Resets all eligibility traces to zero.
        """

        self.cl_visited_pairs.clear()
        for x in self.di_e_table.keys():
            self.di_e_table[x] = np.zeros((self.in_action_size,))
        
    def select_action(self, x):
        """
        Employs and epsilon-greedy action selection and returns the selected action.
        If the eligibility traces are not initialized for the given x,
        this method initializes them first.

        Parameters
        ----------
        x : numpy array
            the x to take an action on

        Returns
        -------
        int
            the selected action for the given x
        """
        # initialize the values
        self.cl_policy.init_q_values(x)
        self.init_e_values(x)

        if np.random.rand() <= self.fl_epsilon:
            self.bo_random_action = True
            return np.random.randint(self.in_action_size)
        self.bo_random_action = False
        return self.cl_policy.get_action(x)
        
    def _dump_policy(self):
        # Configuring eligibility table dictionary to dump with policy
        di_to_dump = {"eligibility_table": {}}
        for o in sorted(self.di_e_table.keys(), key=cmp_to_key(QPolicy._sort_q_table_keys)):
            di_to_dump["eligibility_table"][str(o)] = {}
            for a in range(self.in_action_size):
                di_to_dump["eligibility_table"][str(o)][str(a)] = str(self.di_e_table[o][a])

        # Dumping the policy and Q values
        self.cl_policy.dump_policy(self.st_debug_folder, self.in_experiment_no, self.in_episode_no, di_to_dump)
