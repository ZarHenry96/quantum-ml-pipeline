import argparse
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

from scipy.stats import ranksums, mannwhitneyu, wilcoxon

import warnings
warnings.filterwarnings('ignore', message='Sample size too small for normal approximation.')


def plot_scatterplot(x_data, y_data, legend_labels, legend_pos, x_label, y_label, title, plot_limit_values, out_file,
                     verbose):
    if out_file.endswith('.pdf'):
        dpi = 300
        width, height = (9, 9) if '\\n' not in title else (10, 10)
    else:
        dpi = 150
        px = 1 / dpi
        width, height = 1350*px, 1350*px

    matplotlib.rcParams['figure.dpi'] = dpi
    fig, ax = plt.subplots(figsize=(width, height))

    title_fontsize = 21
    axis_label_fontsize = 20
    tick_label_fontsize = 20
    legend_fontsize = 17.5
    plot_error_bars = False

    markers = ['o', '*', '+', '.', 'x', 's', 'd']
    base_size = 100
    marker_sizes = [base_size, base_size+2, base_size+1, base_size+1, base_size, base_size, base_size]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    if verbose:
        print('\n\n{}\n{} vs {}'.format(title, x_label, y_label))

    for i, (input_x_vals, input_y_vals) in enumerate(zip(x_data, y_data)):
        legend_label = legend_labels[i] if legend_labels is not None else None

        x_vals, x_stds, y_vals, y_stds = input_x_vals, None, input_y_vals, None
        if hasattr(input_x_vals[0], '__len__'):
            x_vals, x_stds = zip(*input_x_vals)
        if hasattr(input_y_vals[0], '__len__'):
            y_vals, y_stds = zip(*input_y_vals)

        ax.scatter(x_vals, y_vals, s=marker_sizes[i % len(marker_sizes)], marker=markers[i % len(markers)],
                   color=colors[i % len(colors)], label=legend_label)
        if plot_error_bars and (x_stds is not None or y_stds is not None):
            ax.errorbar(x_vals, y_vals, xerr=x_stds, yerr=y_stds, ls="None", capsize=3,
                        alpha=0.4, antialiased=True, color=colors[i % len(colors)])

        if verbose:
            x_better = [j for j in range(0, len(x_vals)) if x_vals[j] > y_vals[j]]
            y_better = [j for j in range(0, len(x_vals)) if x_vals[j] < y_vals[j]]
            equal_xy = [j for j in range(0, len(x_vals)) if x_vals[j] == y_vals[j]]
            print('')
            if legend_label is not None:
                print(legend_label)
            print('X better: {}'.format(len(x_better)))
            print('Y better: {}'.format(len(y_better)))
            print('Equal XY: {}'.format(len(equal_xy)))

    l_limit, u_limit = plot_limit_values
    ax.plot([l_limit, u_limit], [l_limit, u_limit], ls="--", c="grey")

    if legend_labels is not None:
        plt.legend(loc=legend_pos, fontsize=legend_fontsize)

    plt.xlim(l_limit, u_limit)
    plt.ylim(l_limit, u_limit)    

    plt.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)

    plt.xlabel(x_label, fontsize=axis_label_fontsize, labelpad=14)
    plt.ylabel(y_label, fontsize=axis_label_fontsize, labelpad=14)
    plt.title(title.replace('\\n', '\n'), fontsize=title_fontsize, pad=17)

    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()


def compute_stats(x_data, y_data, legend_labels, stats_test, stats_out_file):
    with open(stats_out_file, 'w') as sof:
        sof.write('test,label,statistic,p_value,significant\n')

        for i, (input_x_vals, input_y_vals) in enumerate(zip(x_data, y_data)):
            label = legend_labels[i].replace(',', ' -') if legend_labels is not None else None

            x_vals, y_vals = input_x_vals, input_y_vals
            if hasattr(input_x_vals[0], '__len__'):
                x_vals, _ = zip(*input_x_vals)
            if hasattr(input_y_vals[0], '__len__'):
                y_vals, _ = zip(*input_y_vals)

            if stats_test == 'ranksums':
                statistic, p_value = ranksums(x_vals, y_vals)
            elif stats_test == 'mannwhitneyu':
                statistic, p_value = mannwhitneyu(x_vals, y_vals)
            elif stats_test == 'wilcoxon':
                if np.any(np.array(x_vals) - np.array(y_vals)):  # there is at least one different element
                    statistic, p_value = wilcoxon(x_vals, y_vals)  # , zero_method='pratt')
                else:
                    statistic, p_value = None, 1
            else:
                statistic, p_value = None, None

            significant = p_value < 0.05

            sof.write('{},{},{},{},{}\n'.format(stats_test, label, statistic, p_value, significant))


def main(x_res_files, x_keys, x_subkeys, y_res_files, y_keys, y_subkeys, folds_aggr_operation, res_files_partitions,
         partition_sizes, legend_labels, legend_pos, x_label, y_label, title, plot_limit_values, out_file,
         stats_test, stats_out_file, verbose):
    if len(x_keys) != len(x_res_files) or len(y_keys) != len(y_res_files):
        print('Provide a proper number of keys for x and y results files!')
        exit(0)

    if res_files_partitions != 1 and (legend_labels is None or len(legend_labels) != res_files_partitions):
        print('Provide a legend with proper length!')
        exit(0)

    if partition_sizes is not None and sum(partition_sizes) != len(x_res_files):
        print('Provide partition sizes that match the number of specified files!')
        exit(0)

    if plot_limit_values is None or len(plot_limit_values) != 2:
        plot_limit_values = [0.39, 1.05]

    # Load results data
    x_results = []
    for x_res_file, x_key in zip(x_res_files, x_keys):
        with open(x_res_file) as rfj:
            x_res = json.load(rfj)[x_key]
            if x_subkeys is not None:
                x_res = {x_subkey: x_res[x_subkey] for x_subkey in x_subkeys}
            if folds_aggr_operation is not None:
                if folds_aggr_operation == 'avg':
                    if hasattr(x_res[list(x_res.keys())[0]][0], '__len__'):
                        x_res = {
                            x_subkey: [[
                                np.mean(list(zip(*x_res[x_subkey]))[0]),
                                np.std(list(zip(*x_res[x_subkey]))[0])
                            ]]
                            for x_subkey in x_res.keys()
                        }
                    else:
                        x_res = {x_subkey: [np.mean(x_res[x_subkey])] for x_subkey in x_res.keys()}
                else:
                    print('Unrecognized fold operation!')
                    exit(0)
            x_results.append(x_res)

    y_results = []
    for y_res_file, y_key in zip(y_res_files, y_keys):
        with open(y_res_file) as rfj:
            y_res = json.load(rfj)[y_key]
            if y_subkeys is not None:
                y_res = {y_subkey: y_res[y_subkey] for y_subkey in y_subkeys}
            if folds_aggr_operation is not None:
                if folds_aggr_operation == 'avg':
                    if hasattr(y_res[list(y_res.keys())[0]][0], '__len__'):
                        y_res = {
                            y_subkey: [[
                                np.mean(list(zip(*y_res[y_subkey]))[0]),
                                np.std(list(zip(*y_res[y_subkey]))[0])
                            ]]
                            for y_subkey in y_res.keys()
                        }
                    else:
                        y_res = {y_subkey: [np.mean(y_res[y_subkey])] for y_subkey in y_res.keys()}
                else:
                    print('Unrecognized fold operation!')
                    exit(0)
            y_results.append(y_res)

    # Prepare data for scatter plots
    if partition_sizes is None:
        partition_size = int(len(x_res_files) / res_files_partitions)
        new_partition_indices = [p*partition_size for p in range(0, res_files_partitions)]
    else:
        new_partition_indices = [sum(partition_sizes[:p]) for p in range(0, res_files_partitions)]
    x_data, y_data = [], []
    for i, (x_res, y_res) in enumerate(zip(x_results, y_results)):
        if i in new_partition_indices:
            x_data.append([])
            y_data.append([])
        x_data[-1] += [val for vals in x_res.values() for val in vals]
        y_data[-1] += [val for vals in y_res.values() for val in vals]

    # Create output directory
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    # Plot the scatter plot
    plot_scatterplot(x_data, y_data, legend_labels, legend_pos, x_label, y_label, title, plot_limit_values, out_file,
                     verbose)

    # Run the selected statistical test (if specified), and save the results
    if stats_test is not None:
        if stats_test in ['ranksums', 'mannwhitneyu', 'wilcoxon']:
            if stats_out_file is None:
                stats_out_file = '{}_stats.csv'.format(os.path.splitext(out_file)[0])
            else:
                os.makedirs(os.path.dirname(stats_out_file), exist_ok=True)
            compute_stats(x_data, y_data, legend_labels, stats_test, stats_out_file)
        else:
            print('Unknown statistical test!')
            exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for plotting scatterplots of results of KNN-Classifier '
                                                 'experiments')
    parser.add_argument('--x-res-files', metavar='x_res_files', type=str, nargs='+', default=None,
                        help='list of results files produced by the collect_multi_exp_results script for the x axis')
    parser.add_argument('--x-keys', metavar='x_keys', type=str, nargs='+', default=None,
                        help='keys of interest, either datasets or method names, in the x-axis results file(s) '
                             '(one per file)')
    parser.add_argument('--x-subkeys', metavar='x_subkeys', type=str, nargs='+', default=None,
                        help='subkeys of interest, either datasets or method names, in the x-axis results file(s)')
    parser.add_argument('--y-res-files', metavar='y_res_files', type=str, nargs='+', default=None,
                        help='list of results files produced by the collect_multi_exp_results script for the y axis')
    parser.add_argument('--y-keys', metavar='y_keys', type=str, nargs='+', default=None,
                        help='keys of interest, either datasets or method names, in the y-axis results file(s) '
                             '(one per file)')
    parser.add_argument('--y-subkeys', metavar='y_subkeys', type=str, nargs='+', default=None,
                        help='subkeys of interest, either datasets or method names, in the y-axis results file(s)')
    parser.add_argument('--apply-folds-aggr-operation', metavar='apply_folds_aggr_operation', type=str, nargs='?',
                        default=None, help='aggregation operation to be applied to the folds accuracy values, supported'
                                           ' values are: avg')
    parser.add_argument('--res-files-partitions', metavar='res_files_partitions', type=int, nargs='?', default=1,
                        help='number of res-files partitions; each partition will have a different color; the default '
                             'number of partitions is 1')
    parser.add_argument('--partition-sizes', metavar='partition_sizes', type=int, nargs='+', default=None,
                        help='list of res-files partition sizes; res-files are split in equally-sized partitions by '
                             'default')
    parser.add_argument('--legend-labels', metavar='legend_labels', type=str, action='store', nargs='+',
                        default=None, help='legend labels, mandatory if res-files-partitions is different from 1 '
                                           '(the # of labels must match the number of partitions)')
    parser.add_argument('--legend-pos', metavar='legend_pos', type=str, nargs='?', default='upper left',
                        help='position of the legend in the plot')
    parser.add_argument('--x-label', metavar='x_label', type=str, action='store', nargs='?',
                        default='', help='label of the x axis')
    parser.add_argument('--y-label', metavar='y_label', type=str, action='store', nargs='?',
                        default='', help='label of the y axis')
    parser.add_argument('--title', metavar='title', type=str, action='store', nargs='?',
                        default='', help='chart title')
    parser.add_argument('--plot-limit-values', metavar='plot_limit_values', type=float, action='store', nargs='+',
                        default=None, help='limit values for the scatter plot (they must be two, i.e., '
                                           'lower and upper); the default values are 0.39 and 1.05.')
    parser.add_argument('--out-file', metavar='out-file', type=str, action='store',
                        default='boxplot.pdf', help='output filename')
    parser.add_argument('--stats-test', metavar='stats_test', type=str, action='store', nargs='?',
                        default=None, help='statistical test to be performed on the scatter plot data, allowed values '
                                           'are: ranksums, mannwhitneyu, wilcoxon. The last one is for paired data.')
    parser.add_argument('--stats-out-file', metavar='stats_out_file', type=str, action='store', nargs='?',
                        default=None, help='statistics output filename. The default value is '
                                           '\'[out-file]\'_stats.csv')
    parser.add_argument('--verbose', dest='verbose', action='store_const', const=True, default=False,
                        help='print some numerical data about the scatter plots')
    args = parser.parse_args()

    if args.x_res_files is not None and args.x_keys is not None \
            and args.y_res_files is not None and args.y_keys is not None:
        main(args.x_res_files, args.x_keys, args.x_subkeys, args.y_res_files, args.y_keys, args.y_subkeys,
             args.apply_folds_aggr_operation, args.res_files_partitions, args.partition_sizes, args.legend_labels,
             args.legend_pos, args.x_label, args.y_label, args.title, args.plot_limit_values, args.out_file,
             args.stats_test, args.stats_out_file, args.verbose)
