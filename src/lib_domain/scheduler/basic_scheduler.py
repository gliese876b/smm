import gym
import random
import numpy as np
from gym import spaces
from ..multi_agent_env import MultiAgentEnv

"""
Basic Scheduler environment
"""
class BasicSchedulerEnv(MultiAgentEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, number_of_processes=8, max_cpu_time=25):
        self.in_number_of_processes = number_of_processes
        self.in_max_cpu_time = max_cpu_time

        # Rewards
        self.fl_default_reward = 0
        self.fl_idle_cpu_reward = -0.5
        self.fl_incorrect_assignment_reward = -1
        self.fl_complete_process_reward = 10


        # State
        # - ID of the process using the CPU [1,...) or 0 for idle CPU
        # - Remaining times of the processes (0 for finished)

        # Actions
        # - make CPU idle (0)
        # - or assign a process to the CPU (assign p1, p2, ...)
        self.cl_action_space = spaces.Discrete(1 + self.in_number_of_processes)

        self.di_current_states = {}
        self.di_dones = {}

        self._initialize_observation_space()


    def _initialize_observation_space(self):
        # Full observability: (CPU status + remaining times of the processes)
        low = np.zeros(1 + self.in_number_of_processes, dtype=int)
        high = np.full(1 + self.in_number_of_processes, self.in_max_cpu_time, dtype=int)
        high[0] = self.in_number_of_processes
        self.cl_observation_space = spaces.Box(low, high, dtype=np.int32)

    def _one_agent_step(self, state, action):
        assert self.cl_action_space.contains(action)

        next_state = np.copy(state)
        reward = self.fl_default_reward
        done = False

        next_state[0] = action
        if action > 0:
            pid = next_state[0]
            if next_state[pid] == 0: # the assigned process is already finished
                reward = self.fl_incorrect_assignment_reward
                next_state[0] = 0 # CPU stays idle
                done = True
            elif next_state[pid] > 0: # the CPU is being used by incomplete process
                next_state[pid] -= 1
                if next_state[pid] == 0:
                    reward = self.fl_complete_process_reward
                    next_state[0] = 0 # CPU becomes idle
        else:
            reward = self.fl_idle_cpu_reward

        if np.array_equal(np.zeros(self.in_number_of_processes, dtype=int), next_state[1:]): # all the processes are finished
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
            self.di_current_states[agent_name] = new_state
            di_rewards[agent_name] = reward
            self.di_dones[agent_name] = done
            di_dones[agent_name] = done

        di_dones["__all__"] = all(self.di_dones.values())

        return self._get_observations_dict(), di_rewards, di_dones, {}

    def _get_initial_state(self):
        initial_state = np.zeros(1 + self.in_number_of_processes, dtype=int)

        for i in range(1, self.in_number_of_processes + 1):
            initial_state[i] = np.random.randint(1, self.in_max_cpu_time + 1)

        return initial_state

    def reset(self):
        for agent_name in self.li_agent_names:
            self.di_current_states[agent_name] = self._get_initial_state()
            self.di_dones[agent_name] = False
        return self._get_observations_dict()

    def _get_one_agent_observation(self, state):
        # Full observability
        return state

    def _get_observations_dict(self):
        obs_dict = {}
        for agent_name in self.li_agent_names:
            obs_dict[agent_name] = self._get_one_agent_observation(self.di_current_states[agent_name])
        return obs_dict

    def get_observation_space_size(self):
        return self.cl_observation_space.shape[0]

    def get_action_space_size(self):
        return self.cl_action_space.n

class BasicSchedulerEnvV1(BasicSchedulerEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(BasicSchedulerEnvV1, self).__init__()

    def _initialize_observation_space(self):
        # Partial observability: show states (active or dead) of all processes if CPU is idle
        low = np.full(self.in_number_of_processes, 0, dtype=int)
        high = np.full(self.in_number_of_processes, 2, dtype=int)
        self.cl_observation_space = spaces.Box(low, high, dtype=np.int32)

    def _get_one_agent_observation(self, state):
        observation = np.full(self.in_number_of_processes, 1, dtype=int) # null info

        if state[0] == 0: # CPU is idle
            for p in range(1, self.in_number_of_processes+1):
                if state[p] == 0:
                    observation[p-1] = 0
                else:
                    observation[p-1] = 2

        return observation
