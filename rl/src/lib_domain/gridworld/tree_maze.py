# This environment is a modified version of OOIs.
#
# Copyright 2017, Vrije Universiteit Brussel (http://vub.ac.be)
#
# OOIs is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OOIs is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

import gym
import sys
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from ..multi_agent_env import MultiAgentEnv

class TreeMazeEnvV1(MultiAgentEnv):
    """
        Tree Maze environment from the paper
        Steckelmacher, Denis, et al. "Reinforcement learning in POMDPs with memoryless options and option-observation initiation sets."
        Thirty-second AAAI conference on artificial intelligence. 2018.

        T-maze with several branches

        This is discrete environment that looks like a tree :

                      ---- 4
                ---- 2
               |      ---- 5
        I ---- 1
               |      ---- 6
                ---- 3
                      ---- 7

        The maze consists of corridors of length S. After each
        corridor is a branch where the agent can go up or down. After going up
        or down, the agent must go right for S steps before reaching the next
        branching point. The tree has a height H. At each episode, a leaf L is
        choosen randomly as a goal state. Reaching the goal gives a reward of 10.
        Each move goves a reward of -0.1.

        The agent is able to observe its position in the current corridor (distance
        between the agent and the next branch), along with an additional binary
        observation, o. During the first H time-steps, o = whether to go up at the
        i-th branch.

        The goal of the environment is to have the agent learn to remember information
        during the first H time-steps, then use it to navigate the maze.
    """

    def __init__(self, size=10, height=3):
        self.size = size
        self.height = height

        self.action_space = spaces.Discrete(3) # Up, Right, Down
        self._initialize_observation_space()
        self._seed()

        self.fl_correct_leaf_reward = 10.0
        self.fl_incorrect_leaf_reward = -1
        self.fl_default_reward = -0.1

        self.di_current_states = {}
        self.di_dones = {}

    def _initialize_observation_space(self):
        """Partial observability: Bit times cell in the corridor times corridor id"""
        low = np.zeros(3, dtype=int)
        high = np.zeros(3, dtype=int)
        high[0] = 1
        high[1] = self.size - 1
        high[2] = self.height
        self.observation_space = spaces.Box(low, high, dtype=np.int32)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _one_agent_step(self, state, action):
        assert self.action_space.contains(action)

        _before_branch, _done, _goal, _timestep = state
        _done = list(_done)
        _goal = list(_goal)

        """Move up or down if possible"""
        if _before_branch == 0 and action in [0, 2]:
            _done.append(action)           # Direction the agent went
            _before_branch = self.size - 1 # New corridor

        """Move right if possible"""
        if _before_branch != 0 and action == 1:
            _before_branch -= 1

        """Compute reward"""
        if _before_branch == 0 and len(_done) == len(_goal):
            # The agent has just reached the end of the corridor leading to a leaf
            reward = self.fl_correct_leaf_reward if _done == _goal else self.fl_incorrect_leaf_reward
            done = True
        else:
            """Other action"""
            reward = self.fl_default_reward
            done = False

        _timestep += 1
        next_state = (_before_branch, tuple(_done), tuple(_goal), _timestep)

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

    def _get_initial_state(self):
        _goal = list(self.np_random.choice([0, 2], self.height))
        _done = []
        _before_branch = self.size - 1
        _timestep = 0

        return (_before_branch, tuple(_done), tuple(_goal), _timestep)

    def reset(self):
        for agent_name in self.li_agent_names:
            self.di_current_states[agent_name] = self._get_initial_state()
            self.di_dones[agent_name] = False
        return self._get_observations_dict()

    def _get_one_agent_observation(self, state):
        observation = np.zeros(3, dtype=int)

        _before_branch, _done, _goal, _timestep = state

        """Produce a binary observation depending on the current time-step"""
        if _timestep < self.height:
            bit = _goal[_timestep]
        else:
            bit = 1 # neutral bit

        observation[0] = bit
        observation[1] = _before_branch
        observation[2] = len(_done)

        return observation

    def _get_observations_dict(self):
        obs_dict = {}
        for agent_name in self.li_agent_names:
            obs_dict[agent_name] = self._get_one_agent_observation(self.di_current_states[agent_name])
        return obs_dict

    def get_observation_space_size(self):
        return self.observation_space.shape[0]

    def get_action_space_size(self):
        return self.action_space.n


class TreeMazeEnvV2(TreeMazeEnvV1):
    """
    Version where agent gets its location in the corridor as either at leftmost, in middle, or at rightmost
    """

    def __init__(self, size=5, height=2):
        super(TreeMazeEnvV2, self).__init__(size, height)

        self.fl_incorrect_leaf_reward = -1
        self.fl_default_reward = -0.01

    def _get_one_agent_observation(self, state):
        observation = np.zeros(3, dtype=int)

        _before_branch, _done, _goal, _timestep = state

        """Produce a binary observation depending on the current time-step"""
        if _timestep < self.height:
            bit = _goal[_timestep]
        else:
            bit = 1 # neutral bit

        observation[0] = bit
        observation[1] = _before_branch if _before_branch == self.size - 1 or _before_branch == 0 else 1
        observation[2] = len(_done)

        return observation


class TreeMazeEnvV3(TreeMazeEnvV2):
    """
    Larger version with the height of 4
    """

    def __init__(self, size=5, height=4):
        super(TreeMazeEnvV3, self).__init__(size, height)
