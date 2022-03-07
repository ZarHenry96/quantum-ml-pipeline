import argparse
import numpy as np
import os
import pandas as pd


def process_exp_results(res_file, print_dir_name=True):
    # Load results .csv file
    results = pd.read_csv(res_file, sep=',')

    # Open output file
    with open(os.path.join(os.path.dirname(res_file), 'results_analysis.txt'), 'w') as results_analysis_file:
        # Compute accuracy per fold and mean accuracy over folds (with std)
        accuracy_per_fold = results.groupby('fold')\
            .apply(lambda x: (x['expected_label'] == x['predicted_label']).sum() / len(x))

        if print_dir_name:
            print('{}\n\nAccuracy per fold:'.format(os.path.dirname(res_file).split(os.sep)[-1]))
        else:
            print('Results\n\nAccuracy per fold:')
        results_analysis_file.write('Accuracy per fold:')
        for fold, accuracy in accuracy_per_fold.iteritems():
            print('\tFold: {} Accuracy: {}'.format(fold, accuracy))
            results_analysis_file.write('\n\tFold: {} Accuracy: {}'.format(fold, accuracy))

        mean_accuracy, accuracy_std = np.mean(accuracy_per_fold), np.std(accuracy_per_fold)
        print('\nMean accuracy: {} +- {}'.format(mean_accuracy, accuracy_std))
        results_analysis_file.write('\n\nMean accuracy: {} +- {}'.format(mean_accuracy, accuracy_std))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for processing the results of experiments on the '
                                                 'KNN-classifier pipeline')
    parser.add_argument('res_dirs', metavar='res_dirs', type=str, nargs='+', default=None,
                        help='list of directories containing the results of the experiments')
    args = parser.parse_args()

    print('\n')
    for res_dir in args.res_dirs:
        process_exp_results(os.path.join(res_dir, 'results.csv'))
        print('\n')
