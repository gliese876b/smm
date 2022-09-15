# -*- coding: utf-8 -*-
from .abstract_memory_controller import AbstractMemoryController


class MemoryControllerFixedWindow(AbstractMemoryController):
    """
    A memory controller implementation where the memory is a
    fixed-sized sequence of previously seen observations.

    Attributes
    ----------

    Methods
    -------
    update_memory()
        updates the memory by pushing the observation to the end
    """

    def __init__(self, configuration, logger):
        """
        Parameters
        ----------
        configuration : Configuration
            an instance to keep configuration details
        logger : ExperimentLogger
            an instance to keep logs of the experiment
        """
        super(MemoryControllerFixedWindow, self).__init__(configuration, logger)
        
    def update_memory(self, observation, action, reward, next_observation, done):
        """
        Updates the memory based on the given transition.
        Pushes the observation at time t to the end of the memory.
        If the memory is full, removes the observation from the
        beginning of it.
        Increments the number of changes on memory.

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
        tu_event = self._form_event(observation, action, reward, next_observation, done)
    
        # memory update
        m = tuple(self.li_memory)
        if self.in_memory_capacity > 0:
            """If the memory is full, pop the event from the beginning"""
            if len(self.li_memory) >= self.in_memory_capacity:
                self.li_memory = self.li_memory[1:]

            self.li_memory.append(tu_event)    
        m_prime = tuple(self.li_memory)
        
        if m != m_prime:
            self.in_number_of_memory_changes += 1

