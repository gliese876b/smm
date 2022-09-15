# -*- coding: utf-8 -*-
import multiprocessing
import time
import sys
import os
from configuration import Configuration
from experiment_runner import ExperimentRunner
from experiment_logger import ExperimentLogger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":
    if len(sys.argv) == 2:
        configuration_file = sys.argv[1]
        main_folder = '/'.join(configuration_file.split('/')[0:-1])

        sys.path.append("..")   # to enable upper level import

        std_logger = ExperimentLogger.get_std_logger()
        configuration = Configuration(configuration_file)
        std_logger.info("The experiments will be started now by the master process with the configuration below\n")
    
        processes = []
        number_of_experiments = int(configuration.get_parameter("[experiment]", "number_of_experiments"))
        
        exp_no = 0
        while exp_no < number_of_experiments:
            if len(multiprocessing.active_children()) < multiprocessing.cpu_count():
                try:
                    experiment_runner = ExperimentRunner(exp_no, configuration, main_folder)
                    
                    experiment_runner.start()
                    processes.append(experiment_runner)
                    exp_no += 1
                except Exception as ex:
                    std_logger.error(str(ex))
                    sys.exit()
            else:
                time.sleep(4)
            
        for p in processes:
            p.join()
        std_logger.info("Experiments are complete.")
