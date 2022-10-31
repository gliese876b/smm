# -*- coding: utf-8 -*-
import sys
import numpy as np
import random
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from ..multi_agent_env import MultiAgentEnv

class ToHEnv(MultiAgentEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # Rewards
        self.fl_invalid_action_reward = 0
        self.fl_default_reward = 0
        self.fl_goal_state_reward = 0

        self.in_number_of_disks = 0
        self.in_number_of_rods = 0

        # Initial states
        self.li_initial_states = []

        # Goal states
        self.li_goal_state = None

        # Actions: left, right, pickup, putdown
        self.action_space = None

        self._initialize_observation_space()

        self.di_current_states = {}
        self.di_dones = {}

    def _one_agent_step(self, ar_state, in_action):
        fl_reward = self.fl_default_reward
        ar_next_state = np.copy(ar_state)
        if in_action == 0:  # left
            in_arm_location = ar_state[-1]
            if in_arm_location > 1:
                ar_next_state[-1] = in_arm_location-1
            else:
                fl_reward = self.fl_invalid_action_reward
        elif in_action == 1:  # right
            in_arm_location = ar_state[-1]
            if in_arm_location < self.in_number_of_rods:
                ar_next_state[-1] = in_arm_location+1
            else:
                fl_reward = self.fl_invalid_action_reward
        elif in_action == 2:  # pickup
            in_arm_location = ar_state[-1]
            ar_disks_in_location = np.where(ar_state[:-1] == in_arm_location)[0]
            ar_disk_at_arm = np.where(ar_state[:-1] == 0)[0]
            if len(ar_disk_at_arm) == 0 and len(ar_disks_in_location) > 0:
                in_disk_to_pickup = ar_disks_in_location.min()
                ar_next_state[in_disk_to_pickup] = 0    # change the disk's place to the arm (0)
            else:
                fl_reward = self.fl_invalid_action_reward
        elif in_action == 3:  # putdown
            in_arm_location = ar_state[-1]
            ar_disks_in_location = np.where(ar_state[:-1] == in_arm_location)[0]
            ar_disk_at_arm = np.where(ar_state[:-1] == 0)[0]
            if len(ar_disk_at_arm) > 0 and (len(ar_disks_in_location) == 0 or (ar_disks_in_location.min() > ar_disk_at_arm[0])):
                ar_next_state[ar_disk_at_arm[0]] = ar_state[-1] # change the disk's place to the current location
            else:
                fl_reward = self.fl_invalid_action_reward
        else:
            self.cl_logger.error("Invalid Action:{}".format(in_action))
            sys.exit(1)

        bo_done = False
        if np.array_equal(ar_next_state, self.li_goal_state):
            fl_reward = self.fl_goal_state_reward
            bo_done = True
        return ar_next_state, fl_reward, bo_done

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
        # Full observability: [disk1 location, disk2 location, ..., arm location]
        low = np.asarray([0]*self.in_number_of_disks + [1])
        high = np.full(self.in_number_of_disks+1, self.in_number_of_rods, dtype=int)
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



'''
    Tower of Hanoi problem with partial observability
'''
class ToHEnvV1(ToHEnv):
    def __init__(self):
        super(ToHEnvV1, self).__init__()

    def _initialize_observation_space(self):
        # Partial observability: binary values of [disk1 is in the location, disk2 is in the location, ..., arm holds a disk]
        low = np.zeros(self.in_number_of_disks+1, dtype=int)
        high = np.full(self.in_number_of_disks+1, 2, dtype=int) # value of 2 is used for goal state observation
        self.observation_space = spaces.Box(low, high, dtype=np.int32)

    def _get_observations_dict(self):
        obs_dict = {}
        for agent_name in self.li_agent_names:
            obs_dict[agent_name] = self._get_one_agent_observation(self.di_current_states[agent_name])
        return obs_dict

    def _get_one_agent_observation(self, ar_state):
        ar_observation = np.zeros(self.in_number_of_disks+1, dtype=int)

        if np.array_equal(ar_state, self.li_goal_state):
            ar_observation = np.full(self.in_number_of_disks+1, 2, dtype=int)
        else:
            # Partial observability
            in_arm_location = ar_state[-1]
            ar_disks_in_location = np.where(ar_state[:-1] == in_arm_location)[0]
            ar_observation[ar_disks_in_location] = 1
            ar_observation[-1] = len(np.where(ar_state[:-1] == 0)[0])

        return ar_observation


'''
    Tower of Hanoi problem with partial observability
'''
class ToHEnvV2(ToHEnv):
    def __init__(self):
        super(ToHEnvV2, self).__init__()

    def _initialize_observation_space(self):
        # Partial observability: binary values of [disk1 is in the location, disk2 is in the location, ..., arm is in the middle rod, arm holds a disk]
        assert(self.in_number_of_rods % 2 == 1)
        low = np.zeros(self.in_number_of_disks+2, dtype=int)
        high = np.full(self.in_number_of_disks+2, 2, dtype=int) # value of 2 is used for goal state observation
        self.observation_space = spaces.Box(low, high, dtype=np.int32)

    def _get_observations_dict(self):
        # Full observability
        obs_dict = {}
        for agent_name in self.li_agent_names:
            obs_dict[agent_name] = self._get_one_agent_observation(self.di_current_states[agent_name])
        return obs_dict

    def _get_one_agent_observation(self, ar_state):
        ar_observation = np.zeros(self.in_number_of_disks+2, dtype=int)

        if np.array_equal(ar_state, self.li_goal_state):
            return np.full(self.in_number_of_disks+1, 2, dtype=int)
        else:
            # Partial observability
            in_arm_location = ar_state[-1]
            ar_disks_in_location = np.where(ar_state[:-1] == in_arm_location)[0]
            ar_observation[ar_disks_in_location] = 1

            ar_observation[-1] = len(np.where(ar_state[:-1] == 0)[0])

            if ( in_arm_location == (self.in_number_of_rods + 1) / 2 ): # arm is in the middle rod
                ar_observation[-2] = np.random.choice(2, 1, p=[0.2, 0.8])   # the shape is visible with 0.8 probability

        return ar_observation



'''
    Tower of Hanoi problem with 3 disks and 3 rods.
    The agent has an arm that enables it to employ actions: left, right, pickup, putdown
    A state is defined as a vector where the first indexes represent the locations of
    the corresponding disks' locations and the last index is for the location of the arm
    Both the disks and the rods are numbered starting from 1.
'''
class ToHd3r3Env(ToHEnv):
    def __init__(self):
        # Rewards
        self.fl_invalid_action_reward = 0
        self.fl_default_reward = 0
        self.fl_goal_state_reward = 10.0

        self.in_number_of_disks = 3
        self.in_number_of_rods = 3

        # Initial states
        self.li_initial_states = [np.array([1] * (self.in_number_of_disks+1)), np.array([2] * (self.in_number_of_disks+1))]

        # Goal states
        self.li_goal_state = np.asarray([self.in_number_of_rods] * (self.in_number_of_disks+1))

        # Actions: left, right, pickup, putdown
        self.action_space = spaces.Discrete(4)

        self._initialize_observation_space()

        self.di_current_states = {}
        self.di_dones = {}

'''
    Tower of Hanoi problem with 3 disks and 3 rods and partial observability.
'''
class ToHd3r3EnvV1(ToHEnvV1):
    def __init__(self):
        # Rewards
        self.fl_invalid_action_reward = -0.01
        self.fl_default_reward = -0.01
        self.fl_goal_state_reward = 10.0

        self.in_number_of_disks = 3
        self.in_number_of_rods = 3

        # Initial states
        self.li_initial_states = [np.array([1] * (self.in_number_of_disks+1)), np.array([2] * (self.in_number_of_disks+1))]

        # Goal states
        self.li_goal_state = np.asarray([self.in_number_of_rods] * (self.in_number_of_disks+1))

        # Actions: left, right, pickup, putdown
        self.action_space = spaces.Discrete(4)

        self._initialize_observation_space()

        self.di_current_states = {}
        self.di_dones = {}

'''
    Tower of Hanoi problem with 3 disks and 3 rods and partial observability with a shape in the middle rod is visible
'''
class ToHd3r3EnvV2(ToHEnvV2):
    def __init__(self):
        # Rewards
        self.fl_invalid_action_reward = 0
        self.fl_default_reward = 0
        self.fl_goal_state_reward = 10.0

        self.in_number_of_disks = 3
        self.in_number_of_rods = 3

        # Initial states
        self.li_initial_states = [np.array([1] * (self.in_number_of_disks+1)), np.array([2] * (self.in_number_of_disks+1))]

        # Goal states
        self.li_goal_state = np.asarray([self.in_number_of_rods] * (self.in_number_of_disks+1))

        # Actions: left, right, pickup, putdown
        self.action_space = spaces.Discrete(4)

        self._initialize_observation_space()

        self.di_current_states = {}
        self.di_dones = {}
