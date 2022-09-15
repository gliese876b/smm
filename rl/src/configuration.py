import os
import sys
from collections import OrderedDict
from experiment_logger import ExperimentLogger


class Configuration:
    def __init__(self, configuration_file):
        filename, file_extension = os.path.splitext(configuration_file)
        self.cl_std_logger = ExperimentLogger.get_std_logger()
        self.di_parameters_planned = OrderedDict()
        self.di_parameters_read = {}
        self.st_configuration_file = ""
        self.st_config_name = ""
        self.bo_is_single_agent = False
        self.bo_grouped_agents = False
        self.li_agent_setting_sections = []
        self.li_learning_parameters_sections = []
        self.di_agent_setting_agent_type = {}
        if file_extension == ".cfg":
            self.st_configuration_file = configuration_file
            self.st_config_name = filename.split('/')[-1]
            self._initialize_parameters_planned()
            self._read_all_files(self.st_configuration_file)
            self._validate_parameters()

    def get_config_name(self):
        return self.st_config_name

    def _is_parameter_planned(self, section, entry):
        try:
            return entry in self.di_parameters_planned[section]
        except Exception:
            return False

    def _add_parameters_planned(self, section, entry):
        if self._is_parameter_planned(section, entry):
            self.cl_std_logger.error("Parameter {} in Section {} has already been planned".format(entry, section))
            sys.exit(1)
        if section not in self.di_parameters_planned.keys():
            self.di_parameters_planned[section] = []

        self.di_parameters_planned[section].append(entry)

    def _initialize_parameters_planned(self):
        # always required problem parameters
        self._add_parameters_planned("[problem]", "domain_name")
        self._add_parameters_planned("[problem]", "episode_end_by_sink_rewards")
        self._add_parameters_planned("[problem]", "episode_end_by_step_limit")

        # always required experiment parameters
        self._add_parameters_planned("[experiment]", "number_of_experiments")
        self._add_parameters_planned("[experiment]", "number_of_episodes")

        # always required debug parameters
        self._add_parameters_planned("[debug]", "active")

        # always required agent parameters
        self._add_parameters_planned("[agent]", "number_of_agents")

    def _read_all_files(self, conf_file):
        with open(conf_file, 'r') as f:
            lines = f.readlines()
            i = 0
            current_section = ""
            while i < len(lines):
                lines[i] = lines[i].strip()
                if lines[i] == "[import]" and self._parse_line(lines[i+1])[0] == "file":
                    parameter, value = self._parse_line(lines[i+1])
                    if parameter == "file":
                        self._read_all_files(os.getcwd() + '/' + value)
                    i += 1
                elif lines[i].startswith('[') and lines[i].endswith(']'):
                    current_section = lines[i]
                elif len(lines[i]) > 0 and not lines[i].startswith("#"):
                    parameter, value = self._parse_line(lines[i])
                    if current_section not in self.di_parameters_read:
                        self.di_parameters_read[current_section] = {}
                    assert (parameter not in self.di_parameters_read[current_section]), "Error:Multiple parameter values!"
                    self.di_parameters_read[current_section][parameter] = value
                i += 1

    def _validate_parameters(self):
        while list(self.di_parameters_planned):
            for st_section in list(self.di_parameters_planned.keys()):
                if not self.di_parameters_planned[st_section]:
                    del self.di_parameters_planned[st_section]
                    continue
                if st_section not in self.di_parameters_read.keys():
                    self.cl_std_logger.error("Planned parameter(s) for section {} is not found".format(st_section))
                    sys.exit(1)
                li_entries = self.di_parameters_planned[st_section][:]  # use the copied list in order to make sure that the traverse is in correct order
                for st_entry in li_entries:
                    if st_entry not in self.di_parameters_read[st_section]:
                        self.cl_std_logger.error("Planned parameter {} for section {} is not found".format(st_entry, st_section))
                        sys.exit(1)
                    else:
                        st_value = self.di_parameters_read[st_section][st_entry]
                        self.di_parameters_planned[st_section].remove(st_entry)
                        if st_section == "[problem]":
                            if st_entry == "episode_end_by_sink_rewards" and st_value == "true":
                                self._add_parameters_planned("[problem]", "sink_rewards")
                            if st_entry == "episode_end_by_step_limit" and st_value == "true":
                                self._add_parameters_planned("[problem]", "step_limit")

                        elif st_section == "[debug]":
                            if st_entry == "active" and st_value == "true":
                                self._add_parameters_planned("[debug]", "experiments")
                                self._add_parameters_planned("[debug]", "episodes")

                        elif st_section == "[agent]":
                            if st_entry == "number_of_agents":
                                if st_value == "1":
                                    self.bo_is_single_agent = True
                                    self._add_parameters_planned("[agent]", "agent_type")
                                    self._add_parameters_planned("[agent]", "init_policy_from_file")
                                else:
                                    self._add_parameters_planned("[agent]", "number_of_agent_groups")
                                    self._add_parameters_planned("[agent]", "agent_settings")
                            if st_entry == "init_policy_from_file" and st_value == "true":
                                self._add_parameters_planned("[agent]", "policy_file_pattern")

                            if st_entry == "agent_type" and self.bo_is_single_agent:
                                self.add_learning_parameters("[learning]", st_value)
                                self.li_learning_parameters_sections.append("[learning]")

                            if st_entry == "number_of_agent_groups":
                                if st_value != "0":
                                    self.bo_grouped_agents = True
                                    number_of_groups = int(st_value)
                                    for i in range(number_of_groups):
                                        formatted_entry = "group{}_agent_ids".format(i+1)
                                        self._add_parameters_planned("[agent]", formatted_entry)

                            if st_entry == "agent_settings":
                                li_agent_settings = self.di_parameters_read[st_section][st_entry].split(' ')
                                for agent_setting in li_agent_settings:
                                    agent_setting_section = "[{}]".format(agent_setting)
                                    self.li_agent_setting_sections.append(agent_setting_section)
                                    self.add_agent_setting_parameters(agent_setting_section)
                                if self.bo_grouped_agents:
                                    for agent_setting in li_agent_settings:
                                        self._add_parameters_planned(st_section, agent_setting + "_groups")
                                else:
                                    for agent_setting in li_agent_settings:
                                        self._add_parameters_planned(st_section, agent_setting + "_agent_ids")

                        elif st_section in self.li_agent_setting_sections:
                            if st_entry == "init_policy_from_file" and st_value == "true":
                                self._add_parameters_planned(st_section, "policy_file_pattern")
                            if st_entry == "agent_type":
                                self.di_agent_setting_agent_type[st_section] = st_value
                            if st_entry == "learning_parameters":
                                learning_parameters_section = "[{}]".format(st_value)
                                if learning_parameters_section not in self.li_learning_parameters_sections:
                                    self.li_learning_parameters_sections.append(learning_parameters_section)
                                    self.add_learning_parameters(learning_parameters_section, self.di_agent_setting_agent_type[st_section])

                        elif st_section in self.li_learning_parameters_sections:
                            if st_entry == "number_of_subpolicies":
                                in_number_of_landmarks = len(self.di_parameters_read[st_section][st_entry].split(' '))
                                in_number_of_subpolicies = int(st_value)
                                if in_number_of_subpolicies != in_number_of_landmarks:
                                    self.cl_std_logger.error("Number of subpolicies must be equal to number of landmarks!")
                                    sys.exit(1)
                                for i in range(in_number_of_subpolicies):
                                    self._add_parameters_planned(st_section, "policy{}_folder".format(i+1))

                            if st_entry == "subagent_type" and st_value == "sarsa_lambda":
                                self._add_parameters_planned(st_section, "lambda")

    def add_learning_parameters(self, learning_parameter_section, agent_type):
        self._add_parameters_planned(learning_parameter_section, "alpha")
        self._add_parameters_planned(learning_parameter_section, "gamma")
        self._add_parameters_planned(learning_parameter_section, "action_selection_strategy")
        self._add_parameters_planned(learning_parameter_section, "epsilon_start")
        self._add_parameters_planned(learning_parameter_section, "epsilon_end")

        if agent_type == "sarsa_lambda_agent":
            self._add_parameters_planned(learning_parameter_section, "lambda")
        elif agent_type == "sarsa_lambda_smm_agent":
            self._add_parameters_planned(learning_parameter_section, "lambda")
            self._add_parameters_planned("[memory]", "memory_actions")
            self._add_parameters_planned("[memory]", "memory_capacity")
            self._add_parameters_planned("[memory]", "event_content")
        elif agent_type == "sarsa_lambda_smm_w_im_agent":
            self._add_parameters_planned(learning_parameter_section, "lambda")
            self._add_parameters_planned("[memory]", "memory_actions")
            self._add_parameters_planned("[memory]", "memory_capacity")
            self._add_parameters_planned("[memory]", "event_content")
            self._add_parameters_planned("[im]", "beta")
            self._add_parameters_planned("[im]", "im_method")
        elif agent_type == "sarsa_lambda_mb_agent":
            self._add_parameters_planned(learning_parameter_section, "lambda")
            self._add_parameters_planned("[memory]", "memory_capacity")
            self._add_parameters_planned("[memory]", "event_content")
            self._add_parameters_planned("[memory]", "memory_controller_type")
        elif agent_type == "vaps1_agent":
            self._add_parameters_planned("[vaps]", "memory_size")
            self._add_parameters_planned("[vaps]", "boltzmann_temperature_start")
            self._add_parameters_planned("[vaps]", "boltzmann_temperature_end")
            self._add_parameters_planned("[vaps]", "b")

    def add_agent_setting_parameters(self, agent_setting_section):
        self._add_parameters_planned(agent_setting_section, "agent_type")
        self._add_parameters_planned(agent_setting_section, "init_policy_from_file")
        self._add_parameters_planned(agent_setting_section, "learning_parameters")

    @staticmethod
    def _parse_line(line):
        parts = line.split('=')
        return parts[0].strip(), parts[1].strip()

    def write_preamble(self, file):
        file.write("# ====== Configuration Contents  ======\n")
        for section in self.di_parameters_read:
            for parameter, key in self.di_parameters_read[section].items():
                file.write("# {} {} = {}\n".format(section, parameter, key))
        file.write('\n')

    def print_preamble(self):
        print("# ====== Configuration Contents  ======")
        for section in self.di_parameters_read:
            for parameter, key in self.di_parameters_read[section].items():
                print("# {} {} = {}".format(section, parameter, key))
        print("")

    def get_possible_parameter(self, section, parameter_name, default_value):
        if not (section in self.di_parameters_read and parameter_name in self.di_parameters_read[section]):
            return default_value
        else:
            return self.di_parameters_read[section][parameter_name]

    def get_parameter(self, section, parameter_name):
        assert (section in self.di_parameters_read and parameter_name in self.di_parameters_read[section]), "Parameter {}-{} not found".format(section, parameter_name)
        return self.di_parameters_read[section][parameter_name]
