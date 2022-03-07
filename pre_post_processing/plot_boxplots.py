import argparse
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

from itertools import cycle
from scipy.stats import ranksums, ttest_rel


def get_common_keys(data):
    common_keys = None
    for data_dict in data:
        if common_keys is None:
            common_keys = data_dict.keys()
        else:
            common_keys = [k for k in common_keys if k in data_dict.keys()]
    return common_keys


def plot_boxplot(box_data, x_ticks_labels, x_label, title, y_limits, out_file):
    matplotlib.rcParams['figure.dpi'] = 300

    width = max(4+len(box_data), 9)
    height = 10
    plt.figure(figsize=(width, height))
    plt.boxplot(box_data)

    plt.xticks(np.arange(1, len(box_data)+1), x_ticks_labels)
    plt.xlabel(x_label, fontsize=13, labelpad=16)
    plt.ylabel('Accuracy', fontsize=13, labelpad=16)

    plt.title(title, fontsize=15, pad=18)
    plt.ylim(y_limits[0], y_limits[1])

    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def set_box_color(bp, color, bg_color, median_color):
    plt.setp(bp['boxes'], color=color, facecolor=bg_color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=median_color, linewidth=1.2)
    plt.setp(bp['fliers'], markeredgecolor=color)


def plot_multi_boxplots(bp_data, x_ticks_labels, legend_labels, x_label, title, y_limits, out_file):
    matplotlib.rcParams['figure.dpi'] = 300
    matplotlib.rcParams['xtick.labelsize'] = 11
    matplotlib.rcParams['ytick.labelsize'] = 11

    width = max(min(3+(len(bp_data))*len(bp_data[0]), 26), 9)
    heigth = 8 if width < 24 else 11
    plt.figure(figsize=(width, heigth))

    bp_width, bp_distance = 0.6, 0.1
    offset_limit = ((bp_width+bp_distance)/2)*(len(bp_data)-1)
    positions = np.linspace(-offset_limit, offset_limit, len(bp_data))

    boxplots = []
    for i, data in enumerate(bp_data):
        bp = plt.boxplot(data, positions=np.array(range(len(data)))*(float(len(bp_data)))+positions[i],
                         widths=bp_width, patch_artist=True)
        boxplots.append(bp)

    # Set colors
    colors = ['#D7191C', '#2C7BB6', '#B6B62C', '#662CB6', '#2EB62C']
    facecolors = ['mistyrose', 'lightcyan', 'lemonchiffon', '#D2B0FF', '#C4FFC4']
    mediancolors = ['maroon', 'midnightblue', 'olive', 'indigo', 'darkgreen']
    for i, bp in enumerate(boxplots):
        set_box_color(bp, colors[i], facecolors[i], mediancolors[i])

    # Set legend
    for i in range(0, len(boxplots)):
        plt.plot([], c=colors[i], label=legend_labels[i])
    plt.legend(fontsize=11)

    # Set other properties
    plt.xticks(range(0, len(x_ticks_labels) * len(bp_data), len(bp_data)), x_ticks_labels)
    plt.xlabel(x_label, fontsize=14, labelpad=16)
    plt.ylabel('Accuracy', fontsize=14, labelpad=16)

    plt.title(title, fontsize=16, pad=18)

    plt.xlim(-(len(bp_data)+0.5), len(x_ticks_labels)*len(bp_data)+0.5)
    plt.ylim(y_limits[0], y_limits[1])

    # Save figure
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def compute_stats(data, stats_axis, labels, legend, stats_output_files, inverted_x_legend=False):
    axes = [0, 1] if stats_axis == 'all' else [int(stats_axis)]

    labels_out = [x.replace('\n', ' ').replace(' + ', '+').replace(' ', '_').replace('=', '_') for x in labels]
    legend_out = None if legend is None \
        else [x.replace('\n', ' ').replace(' + ', '+').replace(' ', '_').replace('=', '_') for x in legend]

    if stats_output_files is None:
        stats_output_files = ['stats_{}_axis.csv'.format(x) for x in axes]
    elif len(axes) != len(stats_output_files):
        print('Provide a proper number of output filenames for the statistics results!')
        exit(0)
    else:
        os.makedirs(os.path.dirname(stats_output_files[0]), exist_ok=True)

    stats_output_data = []
    if 0 in axes:
        stats_output_data.append([])
        for bp_index, bp_list in enumerate(data):
            for i in range(0, len(bp_list)-1):
                for j in range(i+1, len(bp_list)):
                    first_term = labels_out[i] if legend_out is None else None
                    second_term = labels_out[j] if legend_out is None else None

                    if first_term is None:
                        first_term = labels_out[i]+'_'+legend_out[bp_index] if not inverted_x_legend \
                                     else legend_out[bp_index]+'_'+labels_out[i]
                    if second_term is None:
                        second_term = labels_out[j]+'_'+legend_out[bp_index] if not inverted_x_legend \
                                      else legend_out[bp_index]+'_'+labels_out[j]

                    stats_output_data[-1].append((
                        first_term,
                        second_term,
                        ranksums(bp_list[i], bp_list[j]),
                        ttest_rel(bp_list[i], bp_list[j]),
                        (bp_list[i], bp_list[j])
                    ))
    if 1 in axes:
        stats_output_data.append([])
        for col_index in range(0, len(data[0])):
            for i in range(0, len(data)-1):
                for j in range(i+1, len(data)):
                    stats_output_data[-1].append((
                        labels_out[col_index]+'_'+legend_out[i] if not inverted_x_legend
                            else legend_out[i]+'_'+labels_out[col_index],
                        labels_out[col_index]+'_'+legend_out[j] if not inverted_x_legend
                            else legend_out[j]+'_'+labels_out[col_index],
                        ranksums(data[i][col_index], data[j][col_index]),
                        ttest_rel(data[i][col_index], data[j][col_index]),
                        (data[i][col_index], data[j][col_index])
                    ))

    for stats_out_data, stats_out_file in zip(stats_output_data, stats_output_files):
        with open(stats_out_file, 'w') as sof:
            sof.write('first_term,second_term,ranksums_p_value,ranksums_significance,'
                      'ttest_p_value,ttest_significance\n')
            for element in stats_out_data:
                sof.write('{},{},{},{},{},{}\n'.format(
                    element[0], element[1],
                    element[2][1], element[2][1] < 0.05,
                    element[3][1], element[3][1] < 0.05
                ))


def main(res_files, keys_of_interest, merge_by, common_subkeys_only, limit_subkeys_to, legend_labels,
         invert_x_and_legend, x_label, title, y_limits, out_file, stats, stats_axis, stats_out_files):
    if keys_of_interest is None:
        print('Provide at least one key of interest (datasets/methods)!')
        exit(0)

    if y_limits is None:
        y_limits = [0, 1.1]
    elif len(y_limits) != 2:
        print('The y limits must be 2!')
        exit(0)

    if stats and (stats_axis not in ['0', '1', 'all']):
        print('Provide a valid stats axis!')
        exit(0)

    # Load results data
    results = []
    for res_file in res_files:
        with open(res_file) as rfj:
            results.append(json.load(rfj))

    # Extract the specified data
    data_to_plot = []
    zip_list = zip(results, cycle(keys_of_interest)) if len(results) > len(keys_of_interest) \
        else zip(cycle(results), keys_of_interest)
    for res, key_of_interest in zip_list:
        if key_of_interest in res:
            data_to_plot.append(res[key_of_interest])
        else:
            print('Key of interest not found in res file!')
            exit(0)

    # Process data
    if len(data_to_plot) >= 2:
        if merge_by is not None:
            if len(data_to_plot) % merge_by != 0:
                print('The length of data_to_plot is not divisible by \'merge_by\'')
                exit(0)

            tmp = []
            for i in range(0, int(len(data_to_plot) / merge_by)):
                merged_data = {}
                for j in range(0, merge_by):
                    merged_data = {**merged_data, **data_to_plot[i * merge_by + j]}
                tmp.append(merged_data)
            data_to_plot = tmp

            common_keys = get_common_keys(data_to_plot)
            for i in range(0, len(data_to_plot)):
                data_to_plot[i] = {x: data_to_plot[i][x] for x in common_keys}
        elif common_subkeys_only:
            common_keys = get_common_keys(data_to_plot)
            for i in range(0, len(data_to_plot)):
                data_to_plot[i] = {x: data_to_plot[i][x] for x in common_keys}
        else:
            all_data = {}
            for i in range(0, len(data_to_plot)):
                all_data = {**all_data, **data_to_plot[i]}
            data_to_plot = [all_data]

    # Filter subkeys if required
    if limit_subkeys_to is not None:
        for i in range(0, len(data_to_plot)):
            data_to_plot[i] = {x: data_to_plot[i][x] for x in limit_subkeys_to}

    # Prepare x_ticks_labels
    x_ticks_labels = []
    max_label_width = 13
    for key in data_to_plot[0].keys():
        items = key.replace('local_simulation', 'simulation')\
                   .replace('rbf', 'gaussian')\
                   .replace('+', '_+_').split('_')
        items_with_spaces = ['']
        for i in range(0, len(items)):
            if len(items_with_spaces[-1]+items[i]) <= max_label_width:
                if items_with_spaces[-1] == '':
                    items_with_spaces[-1] = items[i]
                else:
                    items_with_spaces[-1] = '{} {}'.format(items_with_spaces[-1], items[i])
            else:
                items_with_spaces.append(items[i])
        x_ticks_labels.append('\n'.join(items_with_spaces))

    # Create output directory
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    # Plot the boxplots
    if len(data_to_plot) == 1:
        values = list(data_to_plot[0].values())
        plot_boxplot(values, x_ticks_labels, x_label, title, y_limits, out_file)

        if stats:
            compute_stats([values], '0', x_ticks_labels, None, stats_out_files)
    else:
        if legend_labels is not None and len(legend_labels) == len(data_to_plot):
            legend = legend_labels
        elif len(keys_of_interest) == len(data_to_plot):
            legend = [
                x.replace('local_simulation', 'simulation').replace('rbf', 'gaussian')
                 .replace('+', '_+_').replace('_', ' ')
                for x in keys_of_interest
            ]
        else:
            legend = None
            print('Provide a legend with proper length!')
            exit(0)

        values = [list(x.values()) for x in data_to_plot]

        if invert_x_and_legend:
            tmp_values = [[] for _ in range(0, len(values[0]))]
            for boxplots in values:
                for i, boxplot in enumerate(boxplots):
                    tmp_values[i].append(boxplot)
            values = tmp_values

            x_ticks_labels, legend = legend, [x.replace('\n', ' ') for x in x_ticks_labels]

        plot_multi_boxplots(values, x_ticks_labels, legend, x_label, title, y_limits, out_file)

        if stats:
            compute_stats(values, stats_axis, x_ticks_labels, legend, stats_out_files,
                          inverted_x_legend=invert_x_and_legend)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for plotting boxplots of results of KNN-Classifier '
                                                 'experiments')
    parser.add_argument('res_files', metavar='res_files', type=str, nargs='+', default=None,
                        help='list of results files produced by the collect_multi_exp_results script')
    parser.add_argument('--keys-of-interest', metavar='keys_of_interest', type=str, nargs='+', default=None,
                        help='keys of interest, either datasets or method names, in the results file(s)')
    parser.add_argument('--merge-by', metavar='merge_by', type=int, nargs='?', default=None,
                        help='number of consecutive results data to merge together (it keeps only the common subkeys)')
    parser.add_argument('--common-subkeys-only', dest='common_subkeys_only', action='store_const', const=True,
                        default=False, help='plot only the data with common \'sub-key\' (it is used only if the '
                                            'merge-by parameter is not set')
    parser.add_argument('--limit-subkeys-to', metavar='limit_subkeys_to', type=str, action='store', nargs='+',
                        default=None, help='limit the subkeys to be plotted to the ones provided. This filtering '
                        'operation is applied after merge-by/common-subkeys-only/plot-all (default).')
    parser.add_argument('--legend-labels', metavar='legend_labels', type=str, action='store', nargs='+',
                        default=None, help='legend labels')
    parser.add_argument('--invert-x-and-legend', dest='invert_x_and_legend', action='store_const', const=True,
                        default=False, help='invert x and legend (multi-boxplots), useful for k-based boxplots')
    parser.add_argument('--x-label', metavar='x_label', type=str, action='store', nargs='?',
                        default='', help='label of the x axis')
    parser.add_argument('--title', metavar='title', type=str, action='store', nargs='?',
                        default='', help='chart title')
    parser.add_argument('--y-limits', metavar='y_limits', type=float, action='store', nargs='+',
                        default=None, help='y limits for the plot')
    parser.add_argument('--out-file', metavar='out-file', type=str, action='store',
                        default='boxplot.pdf', help='output filename')
    parser.add_argument('--stats', dest='stats', action='store_const', const=True,
                        default=False, help='compute statistics on data_to_plot')
    parser.add_argument('--stats-axis', metavar='stats_axis', type=str, action='store', default='0',
                        help='axis on which to compute the statistics, allowed values are: 0, 1, all')
    parser.add_argument('--stats-out-files', metavar='stats-out-files', type=str, action='store', nargs='+',
                        default=None, help='statistics output filename(s), two filenames are required if stats-axis is '
                        'set to \'all\' (0 and 1 axis)')
    args = parser.parse_args()

    if args.res_files is not None:
        main(args.res_files, args.keys_of_interest, args.merge_by, args.common_subkeys_only, args.limit_subkeys_to,
             args.legend_labels, args.invert_x_and_legend, args.x_label, args.title, args.y_limits, args.out_file,
             args.stats, args.stats_axis, args.stats_out_files)
