# -*- coding: utf-8 -*-
import sys
import numpy as np
import random
from gym import spaces
from ..multi_agent_env import MultiAgentEnv


class LoadUnloadEnv(MultiAgentEnv):
    """
    A class to simulate Load/Unload problem given in
    'Meuleau, N., Peshkin, L., Kim, K. E., & Kaelbling, L. P. (2013).
    Learning finite-state controllers for partially observable environments.'

    Attributes
    ----------
    metadata : dict
        a dictionary to keep meta data
    grid_size : tuple
        a tuple to keep the size of the grid as width and height
    grid : list
        a list containing strings for each row of the grid
        where 'X' represents an obstacle/wall and '_' represents an empty cell
    fl_invalid_action_reward : float
        the reward for taking an invalid action such as moving into a wall
    fl_default_reward : float
        the reward for regular actions
    fl_unloading_reward : float
        the reward for reaching the unload station after being loaded
    fl_intended_direction_prob : float
        the probability of achieving the intended action
    li_initial_states : list
        a list of initial states kept as tuples
    cl_action_space : spaces.Discrete
        the action space for the actions; east and west
    di_current_states : dict
        a dictionary to keep the current states of each agent
    di_dones : dict
        a dictionary to keep 'done' values for each agent
    """

    metadata = {'render.modes': ['human']}

    grid_size = (5, 1)
    grid = ["_____"]

    def __init__(self):
        """Rewards"""
        self.fl_invalid_action_reward = 0
        self.fl_default_reward = 0
        self.fl_unloading_reward = 1.0

        """The probability of achieving the intended action"""
        self.fl_intended_direction_prob = 1.0

        """Initial states"""
        self.li_initial_states = [(0, 0)]

        """Actions: e w"""
        self.cl_action_space = spaces.Discrete(2)

        self._initialize_observation_space()

        self.di_current_states = {}
        self.di_dones = {}

    def _is_state_available(self, state):
        """
        Returns if the given state is a valid state or not

        Parameters
        ----------
        state: tuple
            the state tuple with the x coordinate of the agent
            and the bit for being loaded/unloaded

        Returns
        -------
        bool
            a boolean that is True if the given state coordinate
            lies in the grid
        """

        x = state[0]
        if x < 0 or x >= self.grid_size[0]:
            return False
        if self.grid[0][x] != '_':
            return False
        return True

    def _one_agent_step(self, state, action):
        """
        Returns the result of taking the given action on the given state.

        Parameters
        ----------
        state : tuple
            the state tuple with the x coordinate of the agent
            and the bit for being loaded/unloaded
        action : int
            the action that the agent takes as an integer

        Returns
        -------
        tuple
            a tuple containing the next state, the reward and a boolean
            for whether the episode is finished or not
        """

        if action not in range(2):
            self.cl_std_logger.error("Invalid Action:{}".format(action))
            sys.exit(1)

        reward = self.fl_default_reward
        possible_states = {state: 0}
        action_effects = [1, -1]     # e w
        for action_index, action_effect in enumerate(action_effects):
            probability = (1.0 - self.fl_intended_direction_prob)
            if action_index == action:  # matched action
                probability = self.fl_intended_direction_prob

            x_ = state[0] + action_effect
            loaded_bit_ = state[1]
            if x_ == 0:
                loaded_bit_ = 0
            elif x_ == self.grid_size[0] - 1:
                loaded_bit_ = 1
            possible_next_state = (x_, loaded_bit_)
            if self._is_state_available(possible_next_state):
                possible_states[possible_next_state] = probability
            else:
                possible_states[state] += probability

        assert(sum(possible_states.values()) == 1.0)
        next_state = random.choices(list(possible_states.keys()), list(possible_states.values()))[0]
        done = False
        if state[1] == 1 and next_state[1] == 0:    # unloading
            reward = self.fl_unloading_reward
            done = True
        elif state == next_state:
            reward = self.fl_invalid_action_reward

        return next_state, reward, done

    def step(self, di_actions):
        if all(self.di_dones.values()):
            self.reset()

        di_rewards = {}
        di_dones = {}
        for agent_name in self.li_agent_names:
            if self.di_dones[agent_name]:
                di_rewards[agent_name] = 0
                di_dones[agent_name] = True
                continue

            action = di_actions[agent_name]
            new_state, reward, done = self._one_agent_step(self.di_current_states[agent_name], action)
            self.di_current_states[agent_name] = new_state
            di_rewards[agent_name] = reward
            self.di_dones[agent_name] = done
            di_dones[agent_name] = done

        di_dones["__all__"] = all(self.di_dones.values())

        return self._get_observations_dict(), di_rewards, di_dones, {}

    def reset(self):
        for agent_name in self.li_agent_names:
            self.di_current_states[agent_name] = random.choice(self.li_initial_states)
            self.di_dones[agent_name] = False
        return self._get_observations_dict()

    def _initialize_observation_space(self):
        """Full observability: (x, l/u)"""
        low = np.zeros(2, dtype=int)
        high = np.zeros(2, dtype=int)
        high[0] = self.grid_size[0] - 1
        high[1] = 1
        self.cl_observation_space = spaces.Box(low, high, dtype=np.int32)

    def get_observation_space_size(self):
        return self.cl_observation_space.shape[0]

    def get_action_space_size(self):
        return self.cl_action_space.n

    def _render(self, mode='human', close=False):
        pass

    def _get_observations_dict(self):
        """
        Returns the states as observations since this is the fully observable version.

        Returns
        -------
        dict
            a dictionary where agent names are mapped to current observations
            kept as numpy arrays
        """
        obs_dict = {}
        for agent_name in self.li_agent_names:
            obs_dict[agent_name] = np.asarray(self.di_current_states[agent_name])
        return obs_dict


class LoadUnloadEnvV1(LoadUnloadEnv):
    """
    A class to simulate a partially observable version
    of the Load/Unload problem. An observation is formed
    as 4 bits representing the presence of an obstacle in
    four navigational directions.
    """

    def __init__(self):
        super(LoadUnloadEnvV1, self).__init__()

    def _initialize_observation_space(self):
        """Partial observability: (wall_north, wall_east, wall_south, wall_west)"""
        low = np.zeros(4, dtype=int)
        high = np.full(4, 2.0, dtype=int)
        self.cl_observation_space = spaces.Box(low, high, dtype=np.int32)

    def _get_one_agent_observation(self, state):
        observation = np.zeros(4, dtype=int)

        """Partial observability"""
        observation[0] = 1
        observation[2] = 1
        """east direction"""
        if state[0] == (self.grid_size[0]-1):
            observation[1] = 1
        """west direction"""
        if state[0] == 0:
            observation[3] = 1

        return observation

    def _get_observations_dict(self):
        obs_dict = {}
        for agent_name in self.li_agent_names:
            obs_dict[agent_name] = self._get_one_agent_observation(self.di_current_states[agent_name])
        return obs_dict

class LoadUnloadEnvV2(LoadUnloadEnvV1):
    """
    A class to simulate a partially observable version
    of the Load/Unload problem where the width of the grid
    varies randomly in [5, 20] in each episode.
    """

    def __init__(self):
        super(LoadUnloadEnvV2, self).__init__()

    def reset(self):
        grid_width = random.choice(range(5, 21))
        self.grid_size = (grid_width, 1)
        self.grid = ['_' * grid_width]

        return super().reset()
