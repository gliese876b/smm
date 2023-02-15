# RL FRAMEWORK FOR MULTIAGENT AND FACTORED PROBLEM SETTING

This is a simplied version of a RL framework we have developed. It provides a simulation framework that an environment
is simulated as a Gym environment and the agent interacts with it. The framework allows to have multiple experiments
with multiple agents but this version is intended for single agent only.


All experiment related parameters are saved in a cfg file. A directory containing cfg files is given to
a bash script which traverses the directory and runs the corresponding experiment. The episodic logs (number of steps
to complete an episode, total reward collected throughout the episode) are saved into a rloutSingle file.
-------------------------------------------------------------------------------------------------------------------------------
## DEPENDENCIES

- The framework runs with Python 3.6+.
- Put the repository to a Linux system and follow the instructions to run.
- Run install.sh in the scripts directory to install the following dependencies
in a virtual environment created with name rl_env under the main folder of the project;
    - [numpy]
    - [gym]
    - [pympler]
    - [sklearn]
    - [matplotlib] (for plotting)
    - [progressbar] (for plotting)
-------------------------------------------------------------------------------------------------------------------------------
## EXPERIMENT FOLDER ORGANIZATION

An example set of experiment folders are given under test_cases directory. The directory is organized into
domains where each domain folder contains two header files; _learning.hdr and _problem_experiment_output.hdr
and cfg files.
- _problem_experiment_output.hdr contains general experiment settings such as the number of experiments to run,
the number of episodes each experiment will take etc.
- _learning.hdr contains some common learning parameters.
- Each cfg file represents a method's experiment and contains method specific parameters.
-------------------------------------------------------------------------------------------------------------------------------
## RUNNING AN EXPERIMENT

To run an experiment whose configuration is stored in a cfg file;
- Activate the virtual environment rl_env,
- Go to scripts folder and run the following command;
  ./traverse-all-subdirectories-and-execute-rl.bash <EXPERIMENT_DIRECTORY>
  where <EXPERIMENT_DIRECTORY> is the directory of config files. For example;
  ./traverse-all-subdirectories-and-execute-rl.bash ../workbench/test_cases/load_unload-v1/
  command will run experiments on partially observable Load/Unload problem with The
  corresponding cfg files.
  This script will visit every cfg file under the folder, run the experiments using multiple
  processes and store the results in a rloutSingle file where each episodic log is written
  row by row. Also, it dumps additional logs into the corresponding debug folder.
-------------------------------------------------------------------------------------------------------------------------------
## PLOTTING RESULTS

You may use plot_generator script to plot the results. This script will require a directory
whose subdirectory contain rloutSingle files, a header name (like TotalReward, Steps etc), and minimum
and maximum values for the Y axis. For example;

  python3 plot_generator.py ../workbench/test_cases/load_unload-v1/sarsa_lambda_smm/ TotalReward -1 11

command will read the header values in all rloutSingle files under the directory, take averages and
confidence intervals by each episode and plot it to a pdf file.
-------------------------------------------------------------------------------------------------------------------------------
