# -*- coding: utf-8 -*-
import sys
import numpy as np
import random
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from ..multi_agent_env import MultiAgentEnv




class CookieEnv(MultiAgentEnv):
    """
    Cookie domain from Icarte, Rodrigo Toro, et al.
    "Learning reward machines for partially observable reinforcement learning."
    Advances in Neural Information Processing Systems. 2019.
    """

    metadata = {'render.modes': ['human']}

    grid_size = (15, 17)
    grid = ["XXXXXXXXXXXXXXX",
            "X___XXXXXXXXXXX",
            "X___XXXXXXXXXXX",
            "X___XXXXXXXXXXX",
            "XX_XXXXXXXXXXXX",
            "XX_XXXXXXXXXXXX",
            "XX_XXXXX______X",
            "XX_XXXXX______X",
            "XX____________X",
            "XX_XXXXX______X",
            "XX_XXXXX______X",
            "XX_XXXXXXXXXXXX",
            "XX_XXXXXXXXXXXX",
            "X___XXXXXXXXXXX",
            "X___XXXXXXXXXXX",
            "X___XXXXXXXXXXX",
            "XXXXXXXXXXXXXXX"]

    def __init__(self):
        self.fl_intended_direction_prob = 1.0

        self.fl_default_reward = 0
        self.fl_goal_reward = 1

        self.tu_button_location = (13, 8)
        self.di_cookie_locations = {1: (1, 1), 2: (3, 1), 3: (1, 15), 4: (3, 15)}

        '''
            State contains;
            - x and y coordinate of the agent
            - cookie id (0 for no cookie, 1-4 for the existing cookie id)
        '''

        # Initial state
        self.tu_initial_state = (2, 8, 0)

        # Actions: n e s w
        self.action_space = spaces.Discrete(4)

        self._initialize_observation_space()

        self.di_previous_states = {}    # to be able to check if the cookie is eaten at a step
        self.di_current_states = {}
        self.di_dones = {}

    def _is_state_available(self, state):
        x, y, _ = state
        if x < 0 or x >= self.grid_size[0]:
            return False
        if y < 0 or y >= self.grid_size[1]:
            return False
        if self.grid[y][x] != '_':
            return False
        return True

    @staticmethod
    def _get_room_id(state):
        x, y, _ = state
        if x in range(1, 4) and y in range(1, 4):  # upper room
            return 0
        elif x in range(1, 8) and y in range(4, 13): # corridor
            return 1
        elif x in range(8, 14) and y in range(6, 11):   # right room
            return 2
        elif x in range(1, 4) and y in range(13, 16): # bottom room
            return 3
        return None

    def _one_agent_step(self, state, action):
        assert self.action_space.contains(action)

        x, y, c = tuple(state)

        possible_states = {state: 1.0 - self.fl_intended_direction_prob}
        action_effects = [(0, -1), (1, 0), (0, 1), (-1, 0)]     # n e s w

        possible_next_state = (x + action_effects[action][0], y + action_effects[action][1], c)
        if self._is_state_available(possible_next_state):
            if possible_next_state not in possible_states.keys():
                possible_states[possible_next_state] = 0
            possible_states[possible_next_state] += self.fl_intended_direction_prob
        else:
            possible_states[state] += self.fl_intended_direction_prob

        assert(sum(possible_states.values()) == 1.0)
        x_, y_, _ = random.choices(list(possible_states.keys()), list(possible_states.values()))[0]

        c_ = c
        if (x_, y_) == self.tu_button_location:
            c_ = np.random.randint(1, 5)
        elif c > 0 and (x_, y_) == self.di_cookie_locations[c]:
            c_ = 0

        next_state = (x_, y_, c_)
        reward = self.fl_default_reward
        done = False
        if c > 0 and c_ == 0:    # the cookie is eaten
            reward = self.fl_goal_reward
            done = True

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
            self.di_previous_states[agent_name] = self.di_current_states[agent_name]
            self.di_current_states[agent_name] = new_state
            di_rewards[agent_name] = reward
            self.di_dones[agent_name] = done
            di_dones[agent_name] = done

        di_dones["__all__"] = all(self.di_dones.values())

        return self._get_observations_dict(), di_rewards, di_dones, {}

    def reset(self):
        for agent_name in self.li_agent_names:
            self.di_previous_states[agent_name] = None
            self.di_current_states[agent_name] = self.tu_initial_state
            self.di_dones[agent_name] = False
        return self._get_observations_dict()

    def _initialize_observation_space(self):
        # Full observability: (x, y, c)
        low = np.zeros(3, dtype=int)
        high = np.zeros(3, dtype=int)
        high[0] = self.grid_size[0] - 1
        high[1] = self.grid_size[1] - 1
        high[2] = max(self.di_cookie_locations.keys())
        self.observation_space = spaces.Box(low, high, dtype=np.int32)

    def get_observation_space_size(self):
        return self.observation_space.shape[0]

    def get_action_space_size(self):
        return self.action_space.n

    def _render(self, mode='human', close=False):
        pass

    def _get_observations_dict(self):
        # Full observability
        obs_dict = {}
        for agent_name in self.li_agent_names:
            obs_dict[agent_name] = np.asarray(self.di_current_states[agent_name])
        return obs_dict

class CookieEnvV1(CookieEnv):

    def __init__(self):
        super(CookieEnvV1, self).__init__()

    def _initialize_observation_space(self):
        # Partial observability: (agent x, agent y, cookie presence)
        low = np.zeros(3, dtype=int)
        high = np.zeros(3, dtype=int)
        high[0] = self.grid_size[0] - 1
        high[1] = self.grid_size[1] - 1
        high[2] = 1
        self.observation_space = spaces.Box(low, high, dtype=np.int32)

    def _get_one_agent_observation(self, state):
        x, y, c = state
        observation = np.zeros(3, dtype=int)
        observation[0] = x
        observation[1] = y

        if c > 0:
            room_agent = CookieEnv._get_room_id(state)
            c_x, c_y = self.di_cookie_locations[c]
            if room_agent == CookieEnv._get_room_id((c_x, c_y, 0)):
                observation[2] = 1

        return observation

    def _get_observations_dict(self):
        obs_dict = {}
        for agent_name in self.li_agent_names:
            obs_dict[agent_name] = self._get_one_agent_observation(self.di_current_states[agent_name])
        return obs_dict

class CookieEnvV2(CookieEnv):

    def __init__(self):
        super(CookieEnvV2, self).__init__()

    def _initialize_observation_space(self):
        # Partial observability: (agent x, agent y, room color, button status, cookie room status, cookie eaten status)
        low = np.zeros(6, dtype=int)
        high = np.zeros(6, dtype=int)
        high[0] = self.grid_size[0] - 1
        high[1] = self.grid_size[1] - 1
        high[2] = 3 # 4 rooms with ids 0-3
        high[3] = 1 # 1 for ending in a state with the button and pressing it
        high[4] = 1 # 1 for ending in a state with the same room as the existing cookie
        high[5] = 1 # 1 for ending in the location with the existing cookie and eating it
        self.observation_space = spaces.Box(low, high, dtype=np.int32)

    def _get_one_agent_observation(self, previous_state, state):
        x, y, c = state
        observation = np.zeros(6, dtype=int)
        observation[0] = x
        observation[1] = y
        observation[2] = CookieEnv._get_room_id(state)

        if (x, y) == self.tu_button_location:
            observation[3] = 1

        if c > 0:
            room_agent = CookieEnv._get_room_id(state)
            c_x, c_y = self.di_cookie_locations[c]
            if room_agent == CookieEnv._get_room_id((c_x, c_y, 0)):
                observation[4] = 1
        elif c == 0 and previous_state:
            _, _, c_prev = previous_state
            if c_prev > 0:
                observation[5] = 1

        return observation

    def _get_observations_dict(self):
        obs_dict = {}
        for agent_name in self.li_agent_names:
            obs_dict[agent_name] = self._get_one_agent_observation(self.di_previous_states[agent_name], self.di_current_states[agent_name])
        return obs_dict


class CookieEnvV3(CookieEnv):

    def __init__(self):
        super(CookieEnvV3, self).__init__()

        self.fl_default_reward = -0.01

    def _initialize_observation_space(self):
        """
            Partial observability:
            WxW size window of cells
            A cell value may have;
            - 0 for empty cell
            - 1 for obstacle/wall (X)
            - 2 for button
            - 3 for cookie
        """
        self.in_window_size = 3
        low = np.zeros(self.in_window_size * self.in_window_size, dtype=int)
        high = np.full(self.in_window_size * self.in_window_size, 3, dtype=int)
        self.observation_space = spaces.Box(low, high, dtype=np.int32)

    def _get_one_agent_observation(self, state):
        x, y, c = state
        observation = np.zeros((self.in_window_size, self.in_window_size), dtype=int)

        for j in range(0, self.in_window_size):
            for i in range(0, self.in_window_size):
                x_ = x - (self.in_window_size - 1) // 2 + i
                y_ = y - (self.in_window_size - 1) // 2 + j
                if self._is_state_available((x_, y_, 0)):
                    if (x_, y_) == self.tu_button_location:
                        observation[j][i] = 2
                    elif c > 0 and (x_, y_) == self.di_cookie_locations[c]:
                        observation[j][i] = 3
                else:
                    observation[j][i] = 1

        return observation.flatten()

    def _get_observations_dict(self):
        obs_dict = {}
        for agent_name in self.li_agent_names:
            obs_dict[agent_name] = self._get_one_agent_observation(self.di_current_states[agent_name])
        return obs_dict
