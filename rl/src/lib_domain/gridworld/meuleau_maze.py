# -*- coding: utf-8 -*-
import sys
import numpy as np
import random
from gym import spaces
from ..multi_agent_env import MultiAgentEnv


class MeuleauMazeEnv(MultiAgentEnv):
    """
    A class to simulate the maze problem given in
    'Meuleau, N., Kim, K., Kaelbling, L. P., & Cassandra, A. R. (1999).
    Solving POMDPs by Searching the Space of Finite Policies.
    UAI’99 Proceedings of the Fifteenth Conference on Uncertainty
    in Artificial Intelligence, 417–426'

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
    fl_goal_state_reward : float
        the reward for reaching a goal state
    fl_intended_direction_prob : float
        the probability of achieving the intended action
    li_initial_states : list
        a list of initial states kept as tuples
    li_goal_states : list
        a list of goal states kept as tuples
    action_space : spaces.Discrete
        the action space for the actions; east and west
    di_current_states : dict
        a dictionary to keep the current states of each agent
    di_dones : dict
        a dictionary to keep 'done' values for each agent
    """

    metadata = {'render.modes': ['human']}

    grid_size = (23, 14)
    grid = ["XXXXXXXXXXXXXXXXXXXXXXX",
            "X_____________________X",
            "X_X_XXXXXXXXXXXXXXX_X_X",
            "X_X_X_____________X_X_X",
            "X_X_X_X_XXXXXXX_X_X_X_X",
            "X_X_X_X_X_____X_X_X_X_X",
            "X_X_X_X_X_X_X_X_X_X_X_X",
            "X_X_X_X_X_XXX_X_X_X_X_X",
            "X_X_X_X_________X_X_X_X",
            "X_X_X_XXXXXXXXXXX_X_X_X",
            "X_X_________________X_X",
            "X_XXXXXXXXXXXXXXXXXXX_X",
            "X_____________________X",
            "XXXXXXXXXXXXXXXXXXXXXXX"]

    def __init__(self):
        # Rewards
        self.fl_invalid_action_reward = -0.01
        self.fl_default_reward = -0.01
        self.fl_goal_state_reward = 5.0

        self.fl_intended_direction_prob = 0.85

        """Initial states"""
        self.li_initial_states = [(11, 12)]

        """Goal states"""
        self.li_goal_states = [(11, 6)]

        """Actions: n e s w"""
        self.action_space = spaces.Discrete(4)

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

        x, y = state
        if x < 0 or x >= self.grid_size[0]:
            return False
        if y < 0 or y >= self.grid_size[1]:
            return False
        if self.grid[y][x] != '_':
            return False
        return True

    def _one_agent_step(self, state, action):
        """
        Returns the result of taking the given action on the given state.

        Parameters
        ----------
        state : tuple
            the state tuple with the x and y coordinate of the agent
        action : int
            the action that the agent takes as an integer

        Returns
        -------
        tuple
            a tuple containing the next state, the reward and a boolean
            for whether the episode is finished or not
        """

        if action not in range(4):
            self.cl_std_logger.error("Invalid Action:{}".format(action))
            sys.exit(1)

        reward = self.fl_default_reward
        state = tuple(state)
        possible_states = {state: 0}
        action_effects = [(0, -1), (1, 0), (0, 1), (-1, 0)]     # n e s w
        for action_index, action_effect in enumerate(action_effects):
            probability = (1.0 - self.fl_intended_direction_prob) / 3.0
            if action_index == action:  # matched action
                probability = self.fl_intended_direction_prob

            possible_next_state = (state[0] + action_effect[0], state[1] + action_effect[1])
            if self._is_state_available(possible_next_state):
                possible_states[possible_next_state] = probability
            else:
                possible_states[state] += probability

        assert(sum(possible_states.values()) == 1.0)
        next_state = random.choices(list(possible_states.keys()), list(possible_states.values()))[0]
        done = False
        if next_state in self.li_goal_states:
            reward = self.fl_goal_state_reward
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
        """Full observability: (x, y)"""
        low = np.zeros(2, dtype=int)
        high = np.zeros(2, dtype=int)
        high[0] = self.grid_size[0] - 1
        high[1] = self.grid_size[1] - 1
        self.observation_space = spaces.Box(low, high, dtype=np.int32)

    def get_observation_space_size(self):
        return self.observation_space.shape[0]

    def get_action_space_size(self):
        return self.action_space.n

    def _render(self, mode='human', close=False):
        pass

    def _get_observations_dict(self):
        obs_dict = {}
        for agent_name in self.li_agent_names:
            obs_dict[agent_name] = np.asarray(self.di_current_states[agent_name])
        return obs_dict


class MeuleauMazeEnvV1(MeuleauMazeEnv):
    """
    A class to simulate a partially observable version
    of the Meuleau Maze problem. An observation is formed
    as 4 bits representing the presence of an obstacle in
    four navigational directions.
    """

    def __init__(self):
        super(MeuleauMazeEnvV1, self).__init__()

    def _initialize_observation_space(self):
        """Partial observability: (wall_north, wall_east, wall_south, wall_west)"""
        low = np.zeros(4, dtype=int)
        high = np.full(4, 2.0, dtype=int)
        self.observation_space = spaces.Box(low, high, dtype=np.int32)

    def _get_one_agent_observation(self, state):
        observation = np.zeros(4, dtype=int)
        if tuple(state) in self.li_goal_states:
            observation = np.full(4, 2, dtype=int)
        else:
            """Partial observability"""
            if state[1] == 0 or self.grid[state[1]-1][state[0]] == 'X':  # north direction
                observation[0] = 1
            if state[0] == (self.grid_size[0]-1) or self.grid[state[1]][state[0]+1] == 'X':   # east direction
                observation[1] = 1
            if state[1] == (self.grid_size[1]-1) or self.grid[state[1]+1][state[0]] == 'X':   # south direction
                observation[2] = 1
            if state[0] == 0 or self.grid[state[1]][state[0]-1] == 'X':  # west direction
                observation[3] = 1

        return observation

    def _get_observations_dict(self):
        obs_dict = {}
        for agent_name in self.li_agent_names:
            obs_dict[agent_name] = self._get_one_agent_observation(self.di_current_states[agent_name])
        return obs_dict


class MeuleauMazeEnvV2(MeuleauMazeEnvV1):
    """
    A class to simulate a smaller, partially observable and stochastic
    version of the Meuleau Maze problem. An observation is formed
    as 4 bits representing the presence of an obstacle in
    four navigational directions and an action ends up in the intended
    direction with 0.85 probability.
    """

    grid_size = (15, 10)
    grid = ["XXXXXXXXXXXXXXX",
            "X_____________X",
            "X_X_XXXXXXX_X_X",
            "X_X_X_____X_X_X",
            "X_X_X_X_X_X_X_X",
            "X_X_X_XXX_X_X_X",
            "X_X_________X_X",
            "X_XXXXXXXXXXX_X",
            "X_____________X",
            "XXXXXXXXXXXXXXX"]

    def __init__(self):
        super(MeuleauMazeEnvV2, self).__init__()

        """Initial states"""
        self.li_initial_states = [(7, 8)]

        """Goal states"""
        self.li_goal_states = [(7, 4)]

        self.fl_intended_direction_prob = 0.85
