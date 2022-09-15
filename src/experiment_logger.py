import logging
import sys
import numpy as np


class ExperimentLogger:
    cl_std_logger = None

    @staticmethod
    def _init_std_logger():
        ExperimentLogger.cl_std_logger = logging.getLogger("std_logger")
        # setting the message format
        message_formatter = logging.Formatter("{%(asctime)s,%(process)d:%(filename)s:%(lineno)d} %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")

        # setting logger's stream handler with message format and  auto flush property
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.flush = sys.stdout.flush
        stdout_handler.setFormatter(message_formatter)
        ExperimentLogger.cl_std_logger.addHandler(stdout_handler)

        ExperimentLogger.cl_std_logger.setLevel(logging.DEBUG)

    @staticmethod
    def get_std_logger():
        if not ExperimentLogger.cl_std_logger:
            ExperimentLogger._init_std_logger()
        return ExperimentLogger.cl_std_logger

    def __init__(self, number_of_episodes=1):
        self.di_log = {}
        self.cl_result_file = None
        self.in_number_of_episodes = number_of_episodes

    def __del__(self):
        if self.cl_result_file:
            self.cl_result_file.flush()
            self.cl_result_file.close()

    def initialize_result_header(self, header):
        self.di_log[header] = np.zeros((self.in_number_of_episodes,))

    def set_result_file(self, result_file):
        self.cl_result_file = result_file

    def set_result_log_value(self, header, episode, value):
        self.di_log[header][episode] = value

    def write_log_headers(self):
        header_list = self.di_log.keys()
        for header in header_list:
            self.cl_result_file.write("{}\t".format(header))
        self.cl_result_file.write('\n')

    def write_episode_logs(self, episode):
        header_list = self.di_log.keys()
        for header in header_list:
            self.cl_result_file.write("{:.5}\t".format(self.di_log[header][episode]))
        self.cl_result_file.write('\n')
        self.cl_result_file.flush()

    def write_result_logs(self, f):
        header_list = self.di_log.keys()
        for header in header_list:
            f.write("{}\t".format(header))
        f.write('\n')
        for episode in range(self.in_number_of_episodes):
            for header in header_list:
                f.write("{:.5}\t".format(self.di_log[header][episode]))
            f.write('\n')
