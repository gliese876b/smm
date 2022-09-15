# -*- coding: utf-8 -*-
import random
import numpy as np
from abc import ABC, abstractmethod


class AbstractAgent(ABC):
    """
    Abstract class for the agents. All of the agents inherits from this class.

    Attributes
    ----------
    in_agent_id : int
        unique id of the agent
    st_learning_setting : str
        learning related settings
    in_experiment_no : int
        the id of the experiment
    cl_configuration : Configuration
        configuration object to hold related experiment parameters
    in_observation_size : int
        the size of an observation vector
    in_action_size : int
        number of actions
    fl_learning_rate : float
        learning rate in [0, 1]
    fl_discount_rate : float
        discount rate in [0, 1]
    st_action_selection_strategy : str
        method to select an action at a time step. Ex: epsilon-greedy
    fl_epsilon : float
        initial epsilon value for epsilon-greedy action selection
    in_number_of_experiments : int
        number of experiments to be taken
    in_number_of_episodes : int
        number of episodes for each experiment
    fl_epsilon_decay : float
        decay value for linear epsilon decay
    bo_debug_active : bool
        boolean value whether debug is active
    li_experiments_to_dump : list
        the list of experiments whose debug files to be dumped
    li_episodes_to_dump : list
        the list of episodes whose debug files to be dumped
    cl_experiment_logger : ExperimentLogger
        logger for keeping experiment-wise logs
    in_episode_no : int
        current episode number
    st_debug_folder : str
        the folder name to dump debug files

    Methods
    -------
    pre_episode()
        prepares the agent before the episode starts
    pre_action(observation)
        returns the action selected by the agent on the given observation
    post_action(observation, action, reward, next_observation, done)
        makes necessary updates on the agent
        based on the experienced transition
    post_episode()
        makes final changes on the agent side after the episode ends
    get_name()
        returns the name of the agent
    get_epsilon()
        returns the current epsilon value
    set_debug_folder()
        sets the given value as the debug folder path
    """

    @abstractmethod
    def __init__(self, agent_id, learning_setting, exp_no, configuration, observation_size, action_size, logger):
        """
        Parameters
        ----------
        agent_id : int
            the ID number of the agent
        learning_setting : str
            learning related settings
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
        self.in_agent_id = agent_id
        self.st_learning_setting = learning_setting
        self.in_experiment_no = exp_no
        self.cl_configuration = configuration
        self.in_observation_size = observation_size
        self.in_action_size = action_size
        self.fl_learning_rate = float(self.cl_configuration.get_parameter(learning_setting, "alpha"))
        self.fl_discount_rate = float(self.cl_configuration.get_parameter(learning_setting, "gamma"))
        self.st_action_selection_strategy = self.cl_configuration.get_parameter(learning_setting, "action_selection_strategy")
        self.fl_epsilon = float(self.cl_configuration.get_parameter(learning_setting, "epsilon_start"))
        epsilon_end = float(self.cl_configuration.get_parameter(learning_setting, "epsilon_end"))
        self.in_number_of_experiments = int(self.cl_configuration.get_parameter("[experiment]", "number_of_experiments"))     
        self.in_number_of_episodes = int(self.cl_configuration.get_parameter("[experiment]", "number_of_episodes"))
        self.fl_epsilon_decay = (self.fl_epsilon - epsilon_end) / (self.in_number_of_episodes-1) if self.in_number_of_episodes > 1 else 0
        
        self.bo_debug_active = self.cl_configuration.get_parameter("[debug]", "active") == "true"
        st_exp_to_dump = self.cl_configuration.get_parameter("[debug]", "experiments")
        st_epi_to_dump = self.cl_configuration.get_parameter("[debug]", "episodes")
        self.li_experiments_to_dump = list(map(lambda x: self.in_number_of_experiments-1 if x == "last" else int(x), st_exp_to_dump.split(' '))) if st_exp_to_dump != "all" else range(self.in_number_of_experiments)
        self.li_episodes_to_dump = list(map(lambda x: self.in_number_of_episodes-1 if x == "last" else int(x), st_epi_to_dump.split(' '))) if st_epi_to_dump != "all" else range(self.in_number_of_episodes)
        
        self.cl_experiment_logger = logger
        
        self.in_episode_no = 0
        self.st_debug_folder = ""

    @abstractmethod
    def pre_episode(self):
        """
        Prepares the agent before the episode starts.
        Any subclass must implement this method.
        """
        pass
    
    @abstractmethod
    def pre_action(self, observation):
        """
        Returns the action selected by the agent on the given observation.
        Any subclass must implement this method.

        Parameters
        ----------
        observation : numpy array
            the current observation

        Returns
        -------
        int
            selected action for the given observation
        """
        pass
        
    @abstractmethod
    def post_action(self, observation, action, reward, next_observation, done):
        """
        Makes necessary updates after a transition.
        Any subclass must implement this method.

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
        pass
    
    @abstractmethod
    def post_episode(self):
        """
        Makes final changes on the agent side after the episode ends.
        Any subclass must implement this method.
        """
        self.in_episode_no += 1
    
    @abstractmethod
    def get_name(self):
        """
        Returns the name of the agent.
        Any subclass must implement this method.

        Returns
        -------
        str
            the name of the agent formed with the agent id
        """
        return "AbstractAgent-{}".format(self.in_agent_id)

    def get_epsilon(self):
        """
        Returns the current value of epsilon.
        Any subclass must implement this method.

        Returns
        -------
        float
            the current value of epsilon
        """
        return self.fl_epsilon
        
    def _is_to_dump(self, episode):
        if self.bo_debug_active and self.in_experiment_no in self.li_experiments_to_dump and episode in self.li_episodes_to_dump:
            return True
        return False
        
    def set_debug_folder(self, debug_folder):
        """
        Sets the given value as the debug folder name.

        Parameters
        ----------
        debug_folder : str
            the name of the debug folder
        """
        self.st_debug_folder = debug_folder
