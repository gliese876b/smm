import gym


class GymWrapper:
    def __init__(self, st_domain):
        self.st_domain = st_domain
        self.cl_first_env = gym.make(st_domain)    # this will be needed to get observation and action space size without being assigned to an agent
        self.di_envs = {}
        self.di_dones = {}
        self.di_prev_observations = {}
        self.li_agent_names = []
        self.cl_std_logger = None

    def _initialize_and_assign_domains(self):
        self.di_envs[self.li_agent_names[0]] = self.cl_first_env
        for agent in self.li_agent_names[1:]:
            self.di_envs[agent] = gym.make(self.st_domain)

    def reset(self):
        if not self.di_envs:
            self._initialize_and_assign_domains()

        di_observations = {}
        for agent in self.li_agent_names:
            di_observations[agent] = self.di_envs[agent].reset()
            self.di_dones[agent] = False

        return di_observations

    def step(self, action_dict):
        di_observations = {}
        di_rewards = {}
        di_dones = {}
        di_info_dicts = {}
        for agent in self.li_agent_names:
            if self.di_dones[agent]:
                di_observations[agent] = self.di_prev_observations[agent]
                di_rewards[agent] = 0
                di_dones[agent] = True
                di_info_dicts[agent] = {}
                continue
            observation, reward, done, di_info = self.di_envs[agent].step(action_dict[agent])
            self.di_prev_observations[agent] = observation
            di_observations[agent] = observation
            di_rewards[agent] = reward
            self.di_dones[agent] = done
            di_dones[agent] = done
            di_info_dicts[agent] = di_info

        di_dones["__all__"] = all(self.di_dones.values())
        return di_observations, di_rewards, di_dones, di_info_dicts

    def set_agent_names(self, names):
        self.li_agent_names = names

    def set_logger(self, logger):
        self.cl_std_logger = logger

    def get_observation_space_size(self):
        return self.cl_first_env.observation_space.shape[0]

    def get_action_space_size(self):
        return self.cl_first_env.action_space.n
