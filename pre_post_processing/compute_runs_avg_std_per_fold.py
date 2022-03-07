import argparse
import json
import numpy as np
import os

from collect_multi_exp_results import MyJSONEncoder
from plot_boxplots import get_common_keys


def main(res_dirs, res_filename, out_dir, out_filename):
    # Load results
    results = []
    for res_dir in res_dirs:
        with open(os.path.join(res_dir, res_filename)) as rf:
            results.append(json.load(rf))

    # Compute stats on fold values
    all_stats = {}
    common_keys = get_common_keys(results)  # common methods
    for key in common_keys:
        key_dicts = [res_dict[key] for res_dict in results]  # method results on all datasets (all runs)
        key_stats = {}
        common_subkeys = get_common_keys(key_dicts)  # common datasets
        for subkey in common_subkeys:
            subkey_values = [key_dict[subkey] for key_dict in key_dicts]  # fold values of method on dataset (all runs)
            subkey_stats = [(np.mean(x), np.std(x)) for x in zip(*subkey_values)]
            key_stats[subkey] = subkey_stats
        all_stats[key] = key_stats

    # Save the output file
    out_dir = res_dirs[0] if out_dir is None else out_dir
    out_filename = res_filename.replace('.json', '_runs_avg_std_per_fold.json') if out_filename is None else out_filename
    with open(os.path.join(out_dir, out_filename), 'w') as out_file:
        out_file.write(json.dumps(all_stats, cls=MyJSONEncoder, ensure_ascii=False, indent=4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for computing the fold-by-fold average/std across runs.')
    parser.add_argument('res_dirs', metavar='res_dirs', type=str, nargs='+', default=None,
                        help='list of directories containing the results files produced by the '
                             'collect_multi_exp_results script.')
    parser.add_argument('--res-filename', metavar='res_filename', type=str, nargs='?',
                        default='results_per_method.json', help='name of the results file produced by the '
                        'collect_multi_exp_results script, the default value is \'results_per_method.json\'.')
    parser.add_argument('--out-dir', metavar='out_dir', type=str, action='store', nargs='?',
                        default=None, help='directory where to store the output file (the first res-dir is used by '
                                           'default).')
    parser.add_argument('--out-filename', metavar='out_filename', type=str, action='store', nargs='?',
                        default=None, help='output filename, the default value is '
                                           '\'[res-filename]_runs_avg_std_per_fold.json\'.')
    args = parser.parse_args()

    main(args.res_dirs, args.res_filename, args.out_dir, args.out_filename)
