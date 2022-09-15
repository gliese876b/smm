# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod


class AbstractMemoryController(ABC):
    """
    A abstract memory controller class where the memory is a
    fixed-sized sequence.

    Attributes
    ----------
    cl_configuration : Configuration
        a configuration object that contains all parameters
    cl_experiment_logger : ExperimentLogger
        logger to keep experimental results
    in_memory_size : int
        size of the memory
    li_memory : list
        memory formed as a sequence of observations
    in_number_of_memory_changes : int
        counter to keep the number of memory changes

    Methods
    -------
    get_memory()
        returns the current memory
    update_memory()
        updates the memory according to the implementation
    clear_memory()
        removes all elements from the memory
    get_state_estimation()
        concatenates the memory and the given observation and returns it
    dump_details()
        dumps memory related details
    get_number_of_memory_updates()
        returns the number of changes on the memory
    """

    @abstractmethod
    def __init__(self, configuration, logger):
        """
        Parameters
        ----------
        configuration : Configuration
            an instance to keep configuration details
        logger : ExperimentLogger
            an instance to keep logs of the experiment
        """
        self.cl_configuration = configuration
        self.cl_experiment_logger = logger

        self.in_memory_capacity = int(self.cl_configuration.get_parameter("[memory]", "memory_capacity"))
        self.st_event_content = self.cl_configuration.get_parameter("[memory]", "event_content")
        self.li_memory = list()
        self.in_number_of_memory_changes = 0
        
    def get_memory(self):
        """
        Returns the current memory

        Returns
        -------
        tuple
            the current memory
        """
        return self.li_memory
    
    def clear_memory(self):
        """
        Makes the memory empty
        Sets the number of memory changes to zero
        """
        self.li_memory = list()
        self.in_number_of_memory_changes = 0
    
    def get_state_estimation(self, observation):
        """
        Forms a state estimation by concatenating the memory
        and the observation and returns it

        Parameters
        ----------
        observation : numpy array
            the observation

        Returns
        -------
        tuple
            the state estimation
        """
        return (tuple(self.li_memory), tuple(observation))
 
    @abstractmethod
    def update_memory(self, observation, action, reward, next_observation, done):
        """
        Updates the memory based on the given transition.
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

    def _form_event(self, observation, action=None, reward=None, next_observation=None, done=None):
        """
        Forms the event to be included in the memory.
        It can be extended to actions, rewards, etc.

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

        Returns
        -------
        tuple
            the event
        """
        tu_event = None
        if 'a' in self.st_event_content:
            tu_event = tuple([tuple(observation), action])
        else:
            tu_event = tuple(observation)
        return tu_event

    def dump_details(self, st_debug_folder, in_experiment_no, in_episode_no):
        """
        Dumps details about the memory controller.
        A subclass may overwrite this method.

        Parameters
        ----------
        st_debug_folder : str
            folder name to dump the debugging details
        in_experiment_no : int
            the number of the experiment to be dumped
        in_episode_no : int
            the number of the episode to be dumped
        """
        pass
        
    def get_number_of_memory_updates(self):
        """
        Returns the number of changes on the memory.

        Returns
        -------
        int
            the number of changes on the memory
        """
        return self.in_number_of_memory_changes
