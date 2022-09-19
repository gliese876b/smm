import sys
import os
import os.path
import math
import re
import numpy as np
from random import sample
from sklearn.utils import resample
import scipy as sp
import scipy.stats
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, UnivariateSpline
from cycler import cycler
from matplotlib.font_manager import FontProperties
import progressbar
import subprocess



plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "text.latex.preamble": [r'\usepackage{dutchcal}', r'\usepackage{amssymb}']}
    )
'''
Error bound modes:
- 0 : 5th and 95th percentile
- 1 : Standard deviation
- 2 : 95% Bootstrap confidence interval
'''
error_bound_mode = 2
'''
Significance test between the baseline and the new method:
- 0 : Off
- 1 : On
'''
significance_test = 0

def read_values(result_file, header, number_of_episodes):
    header_values = np.zeros((number_of_episodes, ))
    with open(result_file, 'r') as f:
        lines = f.readlines()
        header_index = -1
        episode_number = 0
        for line in lines:
            line = line.strip()
            if not line.startswith("#") and header in line:
                header_index = line.split('\t').index(header)
                continue

            if episode_number >= number_of_episodes:
                break

            if header_index >= 0:
                header_values[episode_number] = float(line.split('\t')[header_index])
                episode_number += 1

        if episode_number < number_of_episodes:
        	print(result_file, "has", episode_number, "episodes (<", number_of_episodes, ")")

    return header_values

def plot_experiment(ax, result_file_dict, header, number_of_episodes, number_of_experiments, plot_legend_name):
    print("*** Number of experiments: ", len(result_file_dict.keys()), "***")
    experiments = []
    if ( len(result_file_dict.keys()) >= 1 ): # Use all rloutSingle files to plot with error bounds
        experiments = result_file_dict.keys()
        if ( "AVG" in experiments ):
            experiments.remove("AVG")

    value_matrix = np.zeros((number_of_episodes, len(experiments)))
    experiment_number = 0

    for exp in sorted(experiments):
        value_matrix[:, experiment_number] = read_values(result_file_dict[exp], header, number_of_episodes)
        experiment_number += 1

    avg_array = np.zeros((number_of_episodes, ))
    p_low_array = np.zeros((number_of_episodes, ))
    p_high_array = np.zeros((number_of_episodes, ))

    bar = progressbar.ProgressBar(maxval=number_of_episodes, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    for e in range(number_of_episodes):
        avg_array[e,] = np.mean(value_matrix[e, :])
        if ( len(experiments) > 1 ):
            if ( error_bound_mode == 0 ):
                p_low_array[e,] = np.percentile(value_matrix[e, :], 5)
                p_high_array[e,] = np.percentile(value_matrix[e, :], 95)
            elif ( error_bound_mode == 1 ):
                std = np.std(value_matrix[e, :])
                p_low_array[e,] = avg_array[e,] - std
                p_high_array[e,] = avg_array[e,] + std
            elif ( error_bound_mode == 2 ):
                l, h = bootstrap_confidence_interval(value_matrix[e, :], sample_size=0.8, iterations=100)
                p_low_array[e,] = l
                p_high_array[e,] = h
            '''
            if ( avg_array[e,] < p_low_array[e,] or avg_array[e,] > p_high_array[e,] ):
                print("Mean out of error bounds:", p_low_array[e,], avg_array[e,], p_high_array[e,])
            '''
        bar.update(e+1)
    bar.finish()
    t = np.arange(1, number_of_episodes+1)

    # Smoothing
    t_smooth, avg_smooth = smooth_data(t, avg_array)

    if ( error_bound_mode >= 0 ):
        p_low_array = smooth_data(t, p_low_array)[1]
        p_high_array = smooth_data(t, p_high_array)[1]

    print("Average(STD) over all episodes: %.2f(%.2f)" % (np.mean(avg_array), np.std(avg_array)))
    print("Average(STD) of last episodes: %.2f(%.2f)" % (np.mean(value_matrix[-1, :]), np.std(value_matrix[-1, :])))

    if ( len(experiments) > 1 ):
        # Shaded error bounds
        if ( error_bound_mode >= 0 ):
            ax.fill_between(t_smooth, p_low_array, p_high_array, alpha=0.2)
        ax.plot(t_smooth, avg_smooth, label=plot_legend_name, linewidth=2.6)

        # Error bars
        #ax.errorbar(t_smooth, avg_smooth, yerr=[p_low_array, p_high_array], fmt='o', elinewidth=1, label=plot_legend_name, capsize=3)
        return (p_low_array).min(), (p_high_array).max()

    ax.plot(t_smooth, avg_smooth, label=plot_legend_name, linewidth=2.6)
    return (avg_array).min(), (avg_array).max()


def smooth_data(x, y):
    s = UnivariateSpline(x, y, k=2, s=0)
    new_x = np.linspace(x.min(), x.max(), 100, endpoint=True)
    new_y = s(new_x)
    return new_x, new_y


def get_experiment_details(result_file):
    print("Reading %s..." % result_file)
    header_set = []
    number_of_experiments = 0
    number_of_episodes = 0
    agent_type = ""
    plot_legend_name = ""
    problem_name = ""
    with open(result_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            index_of_eq = line.index('=') if '=' in line else -1
            if ( line ):
                if ( "number_of_experiments" in line ):
                    number_of_experiments = int(line[index_of_eq+1:].strip())
                elif ( "number_of_episodes" in line ):
                    number_of_episodes = int(line[index_of_eq+1:].strip())
                elif ( "plot_legend_name" in line ):
                    plot_legend_name = line[index_of_eq+1:].strip()
                elif ( "problem_name" in line ):
                    problem_name = line[index_of_eq+1:].strip()
                elif ( "agent_type" in line ):
                    agent_type = line[index_of_eq+1:].strip()
                if ( not line.startswith('#') ):
                    header_set = line.split('\t')
                    break
    return header_set, number_of_episodes, number_of_experiments, plot_legend_name, problem_name, agent_type

def mean_confidence_interval(data, confidence=0.95):
    n = len(data)
    m, se = np.mean(data), scipy.stats.sem(data)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, h


def bootstrap_confidence_interval(dataset, confidence=0.95, iterations=10000, sample_size=1.0, statistic=np.mean):
    """
    Bootstrap the confidence intervals for a given sample of a population
    and a statistic.

    Args:
        dataset: A list of values, each a sample from an unknown population
        confidence: The confidence value (a float between 0 and 1.0)
        iterations: The number of iterations of resampling to perform
        sample_size: The sample size for each of the resampled (0 to 1.0
                     for 0 to 100% of the original data size)
        statistic: The statistic to use. This must be a function that accepts
                   a list of values and returns a single value.

    Returns:
        Returns the upper and lower values of the confidence interval.
    """
    stats = list()
    n_size = int(len(dataset) * sample_size)

    for _ in range(iterations):
        # Sample (with replacement) from the given dataset
        sample = resample(dataset, n_samples=n_size)
        # Calculate user-defined statistic and store it
        stat = statistic(sample)
        stats.append(stat)

    # Sort the array of per-sample statistics and cut off ends
    ostats = sorted(stats)
    lval = np.percentile(ostats, ((1 - confidence) / 2) * 100)
    uval = np.percentile(ostats, (confidence + ((1 - confidence) / 2)) * 100)

    return (lval, uval)

def strip_experiment_name(file_name, file_extension):
    if ( file_extension == "rloutSingle" ):
        return file_name[:file_name.rfind('_')], file_name[file_name.rfind('_')+1:]
    return file_name, "AVG"


def calculate_p_value(distribution1, distribution2, difference):
    s1 = np.std(distribution1)
    s2 = np.std(distribution2)

    n1 = float(distribution1.shape[0])
    n2 = float(distribution2.shape[0])

    m1 = np.mean(distribution1)
    m2 = np.mean(distribution2)

    se = np.sqrt(np.power(s1, 2) / n1 + np.power(s2, 2) / n2)

    df = np.power(np.power(s1, 2) / n1 + np.power(s2, 2) / n2, 2) / ( np.power(np.power(s1, 2) / n1, 2) / (n1-1) + np.power(np.power(s2, 2) / n2, 2) / (n2-1) )

    t = ((m1 - m2) - difference) / se

    p = 1 - sp.stats.t.cdf(t,df=df)

    return p

def test_for_improvement(result_files, exp_details, original_exp_name, improved_exp_name, significance_level):
    '''
        Hypothesis testing based on
        - https://stattrek.com/hypothesis-test/difference-in-means.aspx,
        - https://machinelearningmastery.com/parametric-statistical-significance-tests-in-python/
    '''
    print("--- Testing the mean difference significance between <%s - %s> ---" % (improved_exp_name, original_exp_name))
    if ( original_exp_name in result_files and len(result_files[original_exp_name].keys()) > 1 and improved_exp_name in result_files and len(result_files[improved_exp_name].keys()) > 1 ):
        number_of_episodes = min(exp_details[original_exp_name][1], exp_details[improved_exp_name][1])

        # Getting the experiments for the original method
        experiments_org = result_files[original_exp_name].keys()
        if ( "AVG" in experiments_org ):
            experiments_org.remove("AVG")

        value_matrix_org = np.zeros((number_of_episodes, len(experiments_org)))
        experiment_number = 0
        for exp in sorted(experiments_org):
            value_matrix_org[:, experiment_number] = read_values(result_files[original_exp_name][exp], header, number_of_episodes)
            experiment_number += 1

        # Getting the experiments for the improved method
        experiments_imp = result_files[improved_exp_name].keys()
        if ( "AVG" in experiments_imp ):
            experiments_imp.remove("AVG")

        value_matrix_imp = np.zeros((number_of_episodes, len(experiments_imp)))
        experiment_number = 0
        for exp in sorted(experiments_imp):
            value_matrix_imp[:, experiment_number] = read_values(result_files[improved_exp_name][exp], header, number_of_episodes)
            experiment_number += 1

        p_value_array = np.zeros((number_of_episodes, ))

        normal_test_p_org = np.zeros((number_of_episodes, ))
        normal_test_p_imp = np.zeros((number_of_episodes, ))

        for e in range(number_of_episodes):
            # Checking if the means are significantly different
            p_value_array[e,] = sp.stats.ttest_ind(value_matrix_org[e, :], value_matrix_imp[e, :], equal_var=False)[1]
            # Checking for normal distributions
            normal_test_p_org[e,] = sp.stats.normaltest(value_matrix_org[e, :])[1]
            normal_test_p_imp[e,] = sp.stats.normaltest(value_matrix_imp[e, :])[1]

        episodes_with_different_means = np.where(p_value_array <= significance_level)[0]

        improved_episode_count = 0
        for e in episodes_with_different_means:
            mean_org = np.mean(value_matrix_org[e, :])
            mean_imp = np.mean(value_matrix_imp[e, :])
            if ( mean_imp < mean_org ):
                improved_episode_count += 1

        print("The number of episodes with significantly (%.2f of significance level) different means = %d (%.2f)" % (significance_level, episodes_with_different_means.shape[0], episodes_with_different_means.shape[0] / float(number_of_episodes)))
        print("The number of episodes with significantly improved = %d (%.2f)" % (improved_episode_count, improved_episode_count / float(number_of_episodes)))

        print("The number of episodes with normal distribution in original method = %d" % (normal_test_p_org[normal_test_p_org <= significance_level].shape[0]))
        print("The number of episodes with normal distribution in improved method = %d" % (normal_test_p_imp[normal_test_p_imp <= significance_level].shape[0]))
    else:
        print("No significance test can be done")



if ( len(sys.argv) == 5 ):
    main_folder_path = sys.argv[1]
    header = sys.argv[2]
    y_min = float(sys.argv[3])
    y_max = float(sys.argv[4])
    result_files = {}
    exp_details = {}

    for file_ in os.listdir(main_folder_path):
        file_name, file_extension = os.path.splitext(file_)
        file_extension = file_extension[1:]
        if file_extension in ["rlout", "rloutSingle"]:
            experiment_name, experiment_no = strip_experiment_name(file_name, file_extension)
            if ( experiment_name not in result_files ):
                result_files[experiment_name] = {}

            result_files[experiment_name][experiment_no] = main_folder_path + file_

    if ( len(result_files) == 0 ):
        print("ERROR: No result file to plot")
        sys.exit()
    problem_name = ""
    number_of_episodes = None
    number_of_experiments = None
    agent_type = ""
    for exp_name, exp_dict in result_files.items():
        exp_details[exp_name] = get_experiment_details(exp_dict[sorted(exp_dict.keys())[0]])
        if ( not problem_name ):
            problem_name = exp_details[exp_name][4]
        elif ( problem_name != exp_details[exp_name][4] ):
            print("Warning: The folder contains result files with different problem names: %s - %s" % (problem_name, exp_details[exp_name][4]))
            #sys.exit()

        if ( not number_of_episodes ):
            number_of_episodes = exp_details[exp_name][1]
        elif ( number_of_episodes != exp_details[exp_name][1] ):
            number_of_episodes = min(number_of_episodes, exp_details[exp_name][1])
            print("Warning: The folder contains result files with different number of episodes, taking min of them", exp_details[exp_name][1])

        if ( not number_of_experiments ):
            number_of_experiments = exp_details[exp_name][2]
        elif ( exp_details[exp_name][2] < number_of_experiments ):
            number_of_experiments = exp_details[exp_name][2]

        if ( not agent_type ):
            agent_type = exp_details[exp_name][5]
        elif ( exp_details[exp_name][5] not in agent_type ):
            agent_type += "-" + exp_details[exp_name][5]

    all_headers = []
    for exp_name in exp_details.keys():
        all_headers += exp_details[exp_name][0]
    all_headers = set(all_headers)

    if header in all_headers:
        print("==== Generating %s Plot ====" % header)
        experiments_with_header = []
        for exp_name in result_files.keys():
            if ( header in exp_details[exp_name][0] ):
                experiments_with_header += [exp_name]

        mpl.rcParams.update({'font.size': 36})
        plt.rc('lines', linewidth=4)
        """List of colors and line styles for each line"""
        # number of memory changes plots
        #plt.rc('axes', prop_cycle=(cycler('color', ['#1f78b4', '#cd00cc', '#ff8000', '#00D8F9', '#984ea3']) + cycler('linestyle', ['-', '-', '-', '-', '-'])))

        # main plots
        plt.rc('axes', prop_cycle=(cycler('color', ['#e41a1c', '#4daf4a', '#1f78b4', '#cd00cc', '#ff8000', '#00D8F9', '#984ea3']) + cycler('linestyle', ['-', '-', '-', '-', '-', '-', '-'])))

        # parameter analysis color cycle
        #plt.rc('axes', prop_cycle=(cycler('color', ['#e41a1c', '#4daf4a', '#1f78b4', '#ff8000', '#00D8F9', '#cd00cc', '#984ea3']) + cycler('linestyle', ['-', '-', '-', '-', '-', '-', '-'])))

        fig = plt.figure(figsize=(16, 10.4), dpi=200)
        ax = fig.add_subplot(111)
        ylim = {}
        ordered_exps = sorted(experiments_with_header)
        ordered_exps = ['basic_scheduler-v1_Sarsa(0.9)-_o_x1-SMM-1', 'basic_scheduler-v1_Sarsa(0.9)-_o_x1-SMM-b1.0-im1-t0-1', 'basic_scheduler-v1_Sarsa(0.9)-_o_x1-SMM-b1.0-im4-t0-1', 'basic_scheduler-v1_Sarsa(0.9)-_o_x1-SMM-b1.0-im3-t0-1']
        print("plotting in the following order:", ordered_exps)
        for exp_name in ordered_exps:
            print("--- Plotting %s file ---" % exp_name)
            try:
                ymin, ymax = plot_experiment(ax, result_files[exp_name], header, number_of_episodes, exp_details[exp_name][2], exp_details[exp_name][3])
            except Exception as e:
                print(str(e))
            if ( "ymin" not in ylim or ymin < ylim["ymin"] ):
                ylim["ymin"] = ymin

            if ( "ymax" not in ylim or ymax > ylim["ymax"] ):
                ylim["ymax"] = ymax

        plt.xlim(0, number_of_episodes)

        if ( y_min == y_max ):
            y_min = math.floor(ylim["ymin"])
            y_max = math.ceil(ylim["ymax"])
        plt.ylim(y_min, y_max)

        header_string = ' '.join(re.findall('[A-Z][^A-Z]*', header))
        ax.set(xlabel='Number of Episodes', ylabel=header_string)

        fontP = FontProperties()
        fontP.set_size(44)

        ax.legend(bbox_to_anchor=(0.5, 1.0), loc="lower center", ncol=4, columnspacing=0.8, handletextpad=0.3, prop=fontP)
        #ax.legend(loc="best", ncol=2, columnspacing=0.6, prop=fontP)

        ax.grid(axis='y', color='lightgray', linestyle='dotted', linewidth=1)
        pdf_file_name = main_folder_path + "/" + "results-" + problem_name + "-" + agent_type + "-" + str(number_of_experiments) + "x" + str(number_of_episodes) + "-" + header + "-[" + str(int(y_min)) + "-" + str(int(y_max)) + "].pdf"
        fig.savefig(pdf_file_name, format="pdf")
        subprocess.run(["pdfcrop", pdf_file_name])

        if ( significance_test ):
            # Test for significance
            baseline_experiments = [exp_name for exp_name in experiments_with_header if "_ABS_NONE" in exp_name]
            for baseline in baseline_experiments:
                exp_detail = baseline[:baseline.index("_ABS_NONE")]
                other_experiments = [exp_name for exp_name in experiments_with_header if exp_detail in exp_name and "_ABS_NONE" not in exp_name]
                for other in sorted(other_experiments):
                    test_for_improvement(result_files, exp_details, baseline, other, 0.05)

    else:
        print("ERROR: Unknown header name")
else:
    print("USAGE: python plotGenerator.py FOLDER_PATH HEADER_NAME Y_MIN Y_MAX")
    print("FOLDER_PATH: The path containing rlout or rloutSingle files")
    print("HEADER_NAME: The column name that the plot is drawn for")
    print("Y_MIN and Y_MAX: The lower and upper limits of the y axis")
