# -*- coding: utf-8 -*-
import multiprocessing
import time
import os
import sys
import datetime
import numpy as np
import gym
from lib_domain import *
from lib_agent.lib_policy import *
from configuration import Configuration
from experiment_logger import ExperimentLogger
from collections import defaultdict
from pympler import asizeof


progress_dot_count = 100


class ExperimentRunner(multiprocessing.Process):
    def __init__(self, exp_no, configuration, main_folder):
        multiprocessing.Process.__init__(self)
        self.in_process_id = os.getpid()
        self.cl_std_logger = ExperimentLogger.get_std_logger()
        self.in_experiment_no = exp_no
        self.cl_configuration = configuration
        self.st_config_name = configuration.get_config_name()

        np.random.seed()

        # setting the experiment folder for given configuration
        if not os.path.isdir(self.st_config_name):
            os.mkdir(self.st_config_name)
        self.st_experiment_folder = main_folder + "/" + self.st_config_name

        self.st_domain = configuration.get_parameter("[problem]", "domain_name")

        if self.st_domain in li_custom_domains:  # check whether wrapper is needed
            self.cl_env = gym.make(self.st_domain)
            self.bo_custom_domain = True
        else:
            self.cl_env = GymWrapper(self.st_domain)
            self.bo_custom_domain = False

        self.cl_env.set_logger(self.cl_std_logger)

        self.in_observation_size = self.cl_env.get_observation_space_size()
        self.in_action_size = self.cl_env.get_action_space_size()
        self.li_headers = ['EpsilonValue', 'NumberOfRewardPeaks', 'Steps', 'TotalReward', 'ElapsedTime']

        self.bo_log_memory_usage = self.cl_configuration.get_possible_parameter("[debug]", "log_memory_usage", "false") == "true"

        if self.bo_log_memory_usage:
            self.li_headers.append('MemoryUsage')

        # Get default parameters
        self.in_number_of_agents = int(self.cl_configuration.get_parameter("[agent]", "number_of_agents"))
        self.bo_is_single_agent = self.in_number_of_agents == 1
        self.di_agent_configurations = {}
        if self.bo_is_single_agent:
            self.di_agent_configurations[1] = {"setting": "[agent]", "learning": "[learning]"}
        else:   # multiagent experiments
            li_agent_settings = self.cl_configuration.get_parameter("[agent]", "agent_settings").split(' ')
            self.in_number_of_agent_groups = int(self.cl_configuration.get_parameter("[agent]", "number_of_agent_groups"))
            self.bo_grouped_agent = self.in_number_of_agent_groups > 0
            if self.bo_grouped_agent:
                self._create_groups()
                self.di_group_settings = {}
                for agent_setting in li_agent_settings:
                    groups_in_setting = self.cl_configuration.get_parameter("[agent]", agent_setting + "_groups").split(' ')
                    for group in groups_in_setting:
                        self.di_group_settings[group] = "[{}]".format(agent_setting)
                for group in self.di_groups.keys():
                    for agent in self.di_groups[group]:
                        agent_setting = self.di_group_settings[group]
                        learning_parameters = "[{}]".format(self.cl_configuration.get_parameter(agent_setting, "learning_parameters"))
                        self.di_agent_configurations[agent] = {"setting": agent_setting, "learning": learning_parameters}
            else:   # individual agents
                for agent_setting in li_agent_settings:
                    agents_in_setting = map(int, self.cl_configuration.get_parameter("[agent]", agent_setting + "_agent_ids").split(' '))
                    formatted_agent_settings = "[{}]".format(agent_setting)
                    for agent in agents_in_setting:
                        learning_parameters = "[{}]".format(self.cl_configuration.get_parameter(formatted_agent_settings, "learning_parameters"))
                        self.di_agent_configurations[agent] = {"setting": formatted_agent_settings, "learning": learning_parameters}

        self.in_number_of_experiments = int(self.cl_configuration.get_parameter("[experiment]", "number_of_experiments"))
        self.in_number_of_episodes = int(self.cl_configuration.get_parameter("[experiment]", "number_of_episodes"))
        self.bo_episode_end_by_step_limit = self.cl_configuration.get_parameter("[problem]", "episode_end_by_step_limit") == "true"
        self.in_step_limit = int(self.cl_configuration.get_parameter("[problem]", "step_limit"))
        self.bo_episode_end_by_sink_rewards = self.cl_configuration.get_parameter("[problem]", "episode_end_by_sink_rewards") == "true"
        self.li_sink_rewards = list(map(float, self.cl_configuration.get_parameter("[problem]", "sink_rewards").split(' ')))

        self.bo_debug_active = self.cl_configuration.get_parameter("[debug]", "active") == "true"

        st_exp_to_dump = self.cl_configuration.get_parameter("[debug]", "experiments")
        st_epi_to_dump = self.cl_configuration.get_parameter("[debug]", "episodes")
        self.li_experiments_to_dump = list(map(lambda x: self.in_number_of_experiments-1 if x == "last" else int(x), st_exp_to_dump.split(' '))) if st_exp_to_dump != "all" else range(self.in_number_of_experiments)
        self.li_episodes_to_dump = list(map(lambda x: self.in_number_of_episodes-1 if x == "last" else int(x), st_epi_to_dump.split(' '))) if st_epi_to_dump != "all" else range(self.in_number_of_episodes)

        # Initialize the loggers
        self.li_experiment_loggers = []
        self._initialize_experiment_loggers()

        # Initialize the agents
        self._initialize_agents()
        self.li_agents_names = self._get_agents_names()
        self.cl_env.set_agent_names(self.li_agents_names)

        # Initialize the main and debug folders of agents
        self.li_agent_folders = []
        self._initialize_agent_folders()
        self.li_debug_folders = []
        self._initialize_agents_debug_folders()

    def _initialize_experiment_loggers(self):
        for i in range(self.in_number_of_agents):
            cl_exp_logger = ExperimentLogger(self.in_number_of_episodes)
            for header in self.li_headers:
                cl_exp_logger.initialize_result_header(header)
            self.li_experiment_loggers.insert(i, cl_exp_logger)

    def _initialize_agent_folders(self):
        for agent_name in self.li_agents_names:
            agent_folder = self.st_experiment_folder + "/" + agent_name
            self.li_agent_folders.append(agent_folder)
            if not os.path.isdir(agent_folder):
                os.mkdir(agent_folder)

    def _initialize_agents_debug_folders(self):
        for i, agent in enumerate(self.li_agents):
            debug_folder = "{}/debug_{}_{}x{}_{}".format(self.li_agent_folders[i], self.st_domain, self.in_number_of_experiments, self.in_number_of_episodes, agent.get_name())
            self.li_debug_folders.append(debug_folder)
            if self.bo_debug_active and not os.path.isdir(debug_folder):
                os.mkdir(debug_folder)
            agent.set_debug_folder(debug_folder)

    def _create_groups(self):
        self.di_groups = {}
        all_ids = []
        for i in range(self.in_number_of_agent_groups):
            group_name = "group" + str(i+1)
            self.di_groups[group_name] = list(map(int, self.cl_configuration.get_parameter("[agent]", group_name + "_agent_ids").split(' ')))
            all_ids.extend(self.di_groups[group_name])

        if len(set(all_ids)) != len(all_ids):
            self.cl_std_logger.error("Some agents are in multiple groups")
            sys.exit(1)

        if len(all_ids) != self.in_number_of_agents:
            self.cl_std_logger.error("Some agents are not assigned to any group")
            sys.exit(1)

    def _make_policy(self, st_agent_type, bo_init_from_file=False, st_policy_file_path=""):
        if st_agent_type == "policy_testing_agent" and not bo_init_from_file:
            self.cl_std_logger.error("Policy must be read from a file for Testing Agent")
            sys.exit(1)
        else:
            policy = QPolicy(self.in_observation_size, self.in_action_size)
            if bo_init_from_file:
                policy.read_policy(st_policy_file_path)
            return policy

    def _create_policies(self):
        self.di_policies = {}
        if self.bo_is_single_agent or self.in_number_of_agent_groups == 0:
            # single agent or individual agents settings (be aware of short-circuit evaluation for getting number of groups)
            self.di_policies["__all__"] = []
            for i in range(self.in_number_of_agents):
                agent_id = i+1
                agent_setting = self.di_agent_configurations[agent_id]["setting"]
                st_agent_type = self.cl_configuration.get_parameter(agent_setting, "agent_type")
                if st_agent_type == "random_agent" or st_agent_type == "dqn_agent" or st_agent_type == "usm_agent":
                    policy = self._make_policy(st_agent_type)
                else:
                    bo_init_policy_from_file = self.cl_configuration.get_parameter(agent_setting, "init_policy_from_file") == "true"
                    if bo_init_policy_from_file:
                        st_policy_file_path = self.cl_configuration.get_parameter(agent_setting, "policy_file_pattern").replace("#", str(self.in_experiment_no))
                        policy = self._make_policy(st_agent_type, bo_init_policy_from_file, st_policy_file_path)
                    else:
                        policy = self._make_policy(st_agent_type)
                self.di_policies["__all__"].append(policy)
        else:
            for group in self.di_groups.keys():
                group_setting = self.di_group_settings[group]
                st_agent_type = self.cl_configuration.get_parameter(group_setting, "agent_type")
                if st_agent_type == "random_agent" or st_agent_type == "dqn_agent":
                    policy = self._make_policy(st_agent_type)
                else:
                    bo_init_policy_from_file = self.cl_configuration.get_parameter(group_setting, "init_policy_from_file") == "true"
                    if bo_init_policy_from_file:
                        st_policy_file_path = self.cl_configuration.get_parameter(group_setting, "policy_file_pattern").replace("#", str(self.in_experiment_no))
                        policy = self._make_policy(st_agent_type, bo_init_policy_from_file, st_policy_file_path)
                    else:
                        policy = self._make_policy(st_agent_type)
                self.di_policies[group] = policy

    def _create_agent(self, agent_type, agent_id, learning_setting, policy, exp_logger):
        if agent_type == "sarsa_lambda_agent":
            from lib_agent.sarsa_lambda_agent import SarsaLambdaAgent
            return SarsaLambdaAgent(agent_id, learning_setting, policy, self.in_experiment_no, self.cl_configuration, self.in_observation_size, self.in_action_size, exp_logger)
        elif agent_type == "sarsa_lambda_smm_agent":
            from lib_agent.sarsa_lambda_smm_agent import SarsaLambdaSMMAgent
            return SarsaLambdaSMMAgent(agent_id, learning_setting, policy, self.in_experiment_no, self.cl_configuration, self.in_observation_size, self.in_action_size, exp_logger)
        elif agent_type == "sarsa_lambda_smm_w_im_agent":
            from lib_agent.sarsa_lambda_smm_w_im_agent import SarsaLambdaSMMwIMAgent
            return SarsaLambdaSMMwIMAgent(agent_id, learning_setting, policy, self.in_experiment_no, self.cl_configuration, self.in_observation_size, self.in_action_size, exp_logger)
        elif agent_type == "sarsa_lambda_mb_agent":
            from lib_agent.sarsa_lambda_memory_based_agent import SarsaLambdaMemoryBasedAgent
            return SarsaLambdaMemoryBasedAgent(agent_id, learning_setting, policy, self.in_experiment_no, self.cl_configuration, self.in_observation_size, self.in_action_size, exp_logger)
        elif agent_type == "vaps1_agent":
            from lib_agent.vaps1_agent import VAPS1Agent
            return VAPS1Agent(agent_id, learning_setting, policy, self.in_experiment_no, self.cl_configuration, self.in_observation_size, self.in_action_size, exp_logger)
        else:
            self.cl_std_logger.error("Agent cannot be created!")
            sys.exit(1)

    def _initialize_agents(self):
        self.li_agents = []
        self._create_policies()
        if self.bo_is_single_agent or self.in_number_of_agent_groups == 0:
            # single agent or individual agents settings (be aware of short-circuit evaluation for getting number of groups)
            for i in range(self.in_number_of_agents):
                agent_id = i + 1
                agent_setting = self.di_agent_configurations[agent_id]["setting"]
                st_agent_type = self.cl_configuration.get_parameter(agent_setting, "agent_type")
                st_learning_setting = self.di_agent_configurations[agent_id]["learning"]
                self.li_agents.append(self._create_agent(st_agent_type, agent_id, st_learning_setting, self.di_policies["__all__"][i], self.li_experiment_loggers[i]))
        else:
            i = 0
            for group in self.di_groups.keys():
                group_setting = self.di_group_settings[group]
                st_agent_type = self.cl_configuration.get_parameter(group_setting, "agent_type")
                st_learning_setting = "[{}]".format(self.cl_configuration.get_parameter(group_setting, "learning_parameters"))
                for agent_id in self.di_groups[group]:
                    self.li_agents.append(self._create_agent(st_agent_type, agent_id, st_learning_setting, self.di_policies[group], self.li_experiment_loggers[i]))
                    i += 1

    def _get_agents_names(self):
        names = []
        for agent in self.li_agents:
            names.append(agent.get_name())

        return names

    @staticmethod
    def _get_agent_memory_usage_in_mb(agent):
        return round(asizeof.asizeof(agent) / (1024 * 1024), 2)

    def run(self):
        start_time = time.process_time()
        if self.in_experiment_no == 0:
            self.cl_configuration.print_preamble()

        # Dump experiment rloutSingle file
        for i, agent_name in enumerate(self.li_agents_names):
            result_file = "{}/{}_{}_{}.rloutSingle".format(self.li_agent_folders[i], self.st_domain, agent_name, self.in_experiment_no)
            f = open(result_file, 'w')
            f.write("# ====== Plot Parameters ======\n")
            f.write("# problem_name = {}\n".format(self.st_domain))
            f.write("# plot_legend_name = {}\n".format(agent_name))
            self.cl_configuration.write_preamble(f)
            self.li_experiment_loggers[i].set_result_file(f)

        self._run_experiment()

        self.cl_std_logger.info("***Experiment {} is finished and lasted {:.4f} seconds***".format(self.in_experiment_no, (time.process_time() - start_time)))

    def _run_experiment(self):
        number_of_done_episodes = 0
        previous_progress = -1

        for i, agent in enumerate(self.li_agents):
            self.li_experiment_loggers[i].write_log_headers()

        for episode in range(self.in_number_of_episodes):

            fl_start_time = time.process_time()
            li_end_times = [0.0] * self.in_number_of_agents
            episode_hists = []
            obs_visits = []
            for i in range(self.in_number_of_agents):
                episode_hists.insert(i, [])
                obs_visits.insert(i, defaultdict(lambda: 0))
            total_rewards = [0] * self.in_number_of_agents

            di_observations = self.cl_env.reset()

            if self.bo_custom_domain:
                di_states = self.cl_env.get_states_dict()

            for agent in self.li_agents:
                agent.pre_episode()

            for i, agent_name in enumerate(self.li_agents_names):
                if self._is_to_dump(episode):
                    observation = di_observations[agent_name]
                    if self.bo_custom_domain:
                        episode_hists[i].append(('t', 'a', 'r', 's', 'o'))
                        state = di_states[agent_name]
                        episode_hists[i].append(('-', '-', '-', state, observation))
                        obs_visits[i][tuple(state)] += 1
                    else:
                        episode_hists[i].append(('t', 'a', 'r', 'o'))
                        episode_hists[i].append(('-', '-', '-', observation))
                        obs_visits[i][tuple(observation)] += 1

            times = [0]*self.in_number_of_agents
            is_done = 0
            di_previous_dones = {}
            for agent_name in self.li_agents_names:
                di_previous_dones[agent_name] = False

            while True:
                di_actions = {}

                for agent in self.li_agents:
                    agent_name = agent.get_name()
                    if not di_previous_dones[agent_name]:
                        di_actions[agent_name] = agent.pre_action(di_observations[agent_name])
                    else:
                        di_actions[agent_name] = 0      # dummy action

                di_next_observations, di_rewards, di_dones, _ = self.cl_env.step(di_actions)

                for i, agent in enumerate(self.li_agents):
                    agent_name = self.li_agents_names[i]
                    if not di_previous_dones[agent_name]:
                        total_rewards[i] += di_rewards[agent_name]
                        agent.post_action(di_observations[agent_name], di_actions[agent_name], di_rewards[agent_name], di_next_observations[agent_name], di_dones[agent_name])
                        times[i] += 1
                di_observations = di_next_observations

                if self.bo_custom_domain:
                    di_states = self.cl_env.get_states_dict()

                # setting the elapsed time for the agents who just finished their episodes.
                for i, agent_name in enumerate(self.li_agents_names):
                    if not di_previous_dones[agent_name] and di_dones[agent_name]:
                        li_end_times[i] = time.process_time()

                if self._is_to_dump(episode):
                    for i, agent_name in enumerate(self.li_agents_names):
                        if not di_previous_dones[agent_name]:
                            if self.bo_custom_domain:
                                episode_hists[i].append((times[i], di_actions[agent_name], di_rewards[agent_name], di_states[agent_name], di_observations[agent_name]))
                                obs_visits[i][tuple(di_states[agent_name])] += 1
                            else:
                                episode_hists[i].append((times[i], di_actions[agent_name], di_rewards[agent_name], di_observations[agent_name]))
                                obs_visits[i][tuple(di_observations[agent_name])] += 1

                max_time = max(times)
                if self.bo_episode_end_by_step_limit and (max_time+1) >= self.in_step_limit:
                    break

                if self.bo_episode_end_by_sink_rewards:
                    for i, agent_name in enumerate(self.li_agents_names):
                        if di_rewards[agent_name] in self.li_sink_rewards:
                            di_dones[agent_name] = True
                            li_end_times[i] = time.process_time()

                    # checking each agent if their status changed after sink reward control
                    all_done = True
                    for agent_name in self.li_agents_names:
                        if not di_dones[agent_name]:
                            all_done = False
                            break

                    if all_done:
                        di_dones["__all__"] = True

                if di_dones["__all__"]:
                    is_done = 1
                    break

                di_previous_dones = di_dones

            # setting the elapsed time for the failing agents
            for i in range(self.in_number_of_agents):
                if li_end_times[i] == 0:
                    li_end_times[i] = time.process_time()

            li_epsilon_values_in_episode = []
            for agent in self.li_agents:
                li_epsilon_values_in_episode.append(agent.get_epsilon())
                agent.post_episode()

            if self._is_to_dump(episode):
                for i in range(self.in_number_of_agents):
                    self._dump_episode_info(self.li_debug_folders[i], episode, episode_hists[i], "rlhist")
                    self._dump_obs_visit_info(self.li_debug_folders[i], episode, obs_visits[i], "rlvisit")

            number_of_done_episodes += is_done

            # Log values
            for i, agent in enumerate(self.li_agents):
                self.li_experiment_loggers[i].set_result_log_value('TotalReward', episode, total_rewards[i])
                self.li_experiment_loggers[i].set_result_log_value('EpsilonValue', episode, li_epsilon_values_in_episode[i])
                self.li_experiment_loggers[i].set_result_log_value('Steps', episode, times[i])
                self.li_experiment_loggers[i].set_result_log_value('NumberOfRewardPeaks', episode, is_done)
                self.li_experiment_loggers[i].set_result_log_value('ElapsedTime', episode, li_end_times[i] - fl_start_time)
                if self.bo_log_memory_usage:
                    self.li_experiment_loggers[i].set_result_log_value('MemoryUsage', episode, self._get_agent_memory_usage_in_mb(agent))

                self.li_experiment_loggers[i].write_episode_logs(episode)

            progress = int((float(episode) / self.in_number_of_episodes) * 100)
            if progress != previous_progress:
                self.cl_std_logger.info("Experiment {} at {}%".format(self.in_experiment_no, progress))
                previous_progress = progress

    def _is_to_dump(self, episode):
        if self.bo_debug_active and self.in_experiment_no in self.li_experiments_to_dump and episode in self.li_episodes_to_dump:
            return True
        return False

    def _dump_episode_info(self, debug_folder, episode, info, ext):
        episode_debug_file = "{}/{}x{}.{}".format(debug_folder, self.in_experiment_no, episode, ext)
        with open(episode_debug_file, 'w') as f:
            if type(info) == list:
                for t in info:
                    for v in t:
                        f.write("{}\t".format(v))
                    f.write("\n")

    def _dump_obs_visit_info(self, debug_folder, episode, info, ext):
        visit_debug_file = "{}/{}x{}.{}".format(debug_folder, self.in_experiment_no, episode, ext)
        with open(visit_debug_file, 'w') as f:
            for k, v in info.items():
                f.write("{}\t:\t{}\n".format(k, v))
