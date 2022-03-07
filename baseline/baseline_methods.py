import argparse
import json
import numpy as np
import os
import pandas as pd
import shutil
import sys
import tarfile

from scipy.spatial.distance import cosine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from knn_plus_classifier import KNNPlusClassifier
from knn_plus_svm import KNNPlusSVM

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import print_dict
from pre_post_processing.process_exp_results import process_exp_results


def preprocess_baseline_config(baseline_config, baseline_config_file_path):
    if not os.path.isabs(baseline_config['dataset']):
        baseline_config['dataset'] = os.path.abspath(
            os.path.join(os.path.dirname(baseline_config_file_path), baseline_config['dataset'])
        )

    if 'class_mapping' in baseline_config and not os.path.isabs(baseline_config['class_mapping']):
        baseline_config['class_mapping'] = \
            os.path.abspath(os.path.join(os.path.dirname(baseline_config_file_path), baseline_config['class_mapping']))

    if not os.path.isabs(baseline_config['res_dir']):
        baseline_config['res_dir'] = os.path.abspath(
            os.path.join(os.path.dirname(baseline_config_file_path), baseline_config['res_dir'])
        )


def feature_normalization(training_set_df, test_set_df):
    features_number = len(training_set_df.columns) - 1

    # Perform attribute normalization by subtracting the minimum and dividing by the range
    training_min = training_set_df.iloc[:, :features_number].min()
    training_range = training_set_df.iloc[:, :features_number].max() - training_set_df.iloc[:, :features_number].min()

    # replace the zero range values with 1
    zero_range_columns = np.nonzero(~(training_range.to_numpy() != 0))[0]
    if len(zero_range_columns) > 0:
        training_range.iloc[zero_range_columns] = [1 for _ in range(0, len(zero_range_columns))]

    test_set_df.iloc[:, :features_number] = (test_set_df.iloc[:, :features_number] - training_min) / training_range
    test_set_df.iloc[:, :features_number] = test_set_df.iloc[:, :features_number].clip(0, 1)

    training_set_df.iloc[:, :features_number] = \
        (training_set_df.iloc[:, :features_number] - training_min) / training_range

    return training_set_df, test_set_df


def run_fold(model, dataset, train, test, fold_index, fold_res_dir, res_file):
    print('Fold {} ...'.format(fold_index), end='')

    # Save training and test data for current fold
    training_set = dataset.iloc[train].copy()
    training_set.reset_index(drop=True, inplace=True)
    training_set_file = os.path.join(fold_res_dir, 'training_data.csv')
    training_set.to_csv(training_set_file, index=False)

    test_set = dataset.iloc[test].copy()
    test_set.reset_index(drop=True, inplace=True)
    test_set_file = os.path.join(fold_res_dir, 'test_instances.csv')
    test_set.to_csv(test_set_file, index=False)

    # Perform feature normalization and save the normalized sets
    training_set, test_set = feature_normalization(training_set, test_set)
    training_set.to_csv(training_set_file.replace('.csv', '_normalized.csv'), index=False)
    test_set.to_csv(test_set_file.replace('.csv', '_normalized.csv'), index=False)

    # Train model and predict labels
    model.fit(training_set.iloc[:, :-1], training_set.iloc[:, -1])
    predicted_labels = model.predict(test_set.iloc[:, :-1])

    # Save results
    with open(os.path.join(fold_res_dir, 'predicted_labels.csv'), 'w') as plf:
        plf.write('predicted_label\n')

        for j, (exp_label, pred_label) in enumerate(zip(test_set.iloc[:, -1], predicted_labels)):
            plf.write('{}\n'.format(pred_label))
            res_file.write('{},{},{},{}\n'.format(fold_index, j, exp_label, pred_label))

    print('\tDone')


def run_random_forest(dataset, training_test_splits, rf_config, res_dir):
    os.makedirs(res_dir, exist_ok=True)
    print('Random forest classifier\n')

    # If the config is not empty, print it
    if rf_config:
        print_dict(rf_config)

    # Save method config file
    with open(os.path.join(res_dir, 'method_config.json'), 'w') as mcf:
        json.dump(rf_config, mcf, ensure_ascii=False, indent=4)

    # Open results file and run folds
    res_filename = os.path.join(res_dir, 'results.csv')
    with open(res_filename, 'w') as res_file:
        res_file.write('fold,test,expected_label,predicted_label\n')

        for i, (train, test) in enumerate(training_test_splits):
            fold_i_res_dir = os.path.join(res_dir, 'fold_{}'.format(i))
            os.makedirs(fold_i_res_dir)

            rf = RandomForestClassifier()
            run_fold(rf, dataset, train, test, i, fold_i_res_dir, res_file)

    print()
    process_exp_results(res_filename, print_dir_name=False)
    print('\n'+'-'*50+'\n')


def run_svm(dataset, training_test_splits, svm_config, method_res_dir):
    for kernel in svm_config['kernels']:
        res_dir = '{}_{}'.format(method_res_dir, kernel)
        os.makedirs(res_dir, exist_ok=True)
        print('SVM with {} kernel\n'.format(kernel))

        svm_kernel_config = dict(svm_config)
        del svm_kernel_config['kernels']
        svm_kernel_config['kernel'] = kernel

        # Print the configuration
        print_dict(svm_kernel_config)
        print()

        # Save method config file
        with open(os.path.join(res_dir, 'method_config.json'), 'w') as mcf:
            json.dump(svm_kernel_config, mcf, ensure_ascii=False, indent=4)

        # Open results file and run folds
        res_filename = os.path.join(res_dir, 'results.csv')
        with open(res_filename, 'w') as res_file:
            res_file.write('fold,test,expected_label,predicted_label\n')

            for i, (train, test) in enumerate(training_test_splits):
                fold_i_res_dir = os.path.join(res_dir, 'fold_{}'.format(i))
                os.makedirs(fold_i_res_dir)

                svm = SVC(kernel=kernel)
                run_fold(svm, dataset, train, test, i, fold_i_res_dir, res_file)

        print()
        process_exp_results(res_filename, print_dir_name=False)
        print('\n'+'-'*50+'\n')


def run_knn(dataset, training_test_splits, knn_config, method_res_dir):
    for metric in knn_config['metrics']:
        res_dir = '{}_{}'.format(method_res_dir, metric)
        os.makedirs(res_dir, exist_ok=True)

        print('KNN with {} distance metric\n'.format(metric))

        knn_metric_config = dict(knn_config)
        del knn_metric_config['metrics']
        knn_metric_config['metric'] = metric

        # Print the configuration
        print_dict(knn_metric_config)
        print()

        # Save method config file
        with open(os.path.join(res_dir, 'method_config.json'), 'w') as mcf:
            json.dump(knn_metric_config, mcf, ensure_ascii=False, indent=4)

        # Open results file and run folds
        res_filename = os.path.join(res_dir, 'results.csv')
        with open(res_filename, 'w') as res_file:
            res_file.write('fold,test,expected_label,predicted_label\n')

            distance_metric = cosine if metric == 'cosine' else metric
            for i, (train, test) in enumerate(training_test_splits):
                fold_i_res_dir = os.path.join(res_dir, 'fold_{}'.format(i))
                os.makedirs(fold_i_res_dir)

                knn = KNeighborsClassifier(n_neighbors=knn_metric_config['k'], metric=distance_metric,
                                           algorithm='brute')
                run_fold(knn, dataset, train, test, i, fold_i_res_dir, res_file)

        print()
        process_exp_results(res_filename, print_dir_name=False)
        print('\n'+'-'*50+'\n')


def run_knn_plus_classifier(dataset, training_test_splits, knn_cl_config, base_res_dir):
    for metric in knn_cl_config['knn_metrics']:
        res_dir_name = 'knn_{}+classifier'.format(metric)
        res_dir = os.path.join(base_res_dir, res_dir_name)
        os.makedirs(res_dir, exist_ok=True)
        print('KNN ({}) + Classifier\n'.format(metric))

        knn_cl_metric_config = dict(knn_cl_config)
        del knn_cl_metric_config['knn_metrics']
        knn_cl_metric_config['knn_metric'] = metric

        # Print the configuration
        print_dict(knn_cl_metric_config)
        print()

        # Save method config file
        with open(os.path.join(res_dir, 'method_config.json'), 'w') as mcf:
            json.dump(knn_cl_metric_config, mcf, ensure_ascii=False, indent=4)

        # Open results file and run folds
        res_filename = os.path.join(res_dir, 'results.csv')
        with open(res_filename, 'w') as res_file:
            res_file.write('fold,test,expected_label,predicted_label\n')

            distance_metric = cosine if metric == 'cosine' else metric
            for i, (train, test) in enumerate(training_test_splits):
                fold_i_res_dir = os.path.join(res_dir, 'fold_{}'.format(i))
                os.makedirs(fold_i_res_dir)

                knn_plus_classifier = KNNPlusClassifier(knn_cl_metric_config['k'], knn_metric=distance_metric,
                                                        res_dir=fold_i_res_dir)
                run_fold(knn_plus_classifier, dataset, train, test, i, fold_i_res_dir, res_file)

                # Compress fold results and delete the original folder
                with tarfile.open('{}.tar.gz'.format(fold_i_res_dir), 'w:gz') as tar:
                    tar.add(fold_i_res_dir, arcname=os.path.basename(fold_i_res_dir))
                shutil.rmtree(fold_i_res_dir)

        print()
        process_exp_results(res_filename, print_dir_name=False)
        print('\n'+'-'*50+'\n')


def run_knn_plus_svm(dataset, training_test_splits, knn_svm_config, base_res_dir):
    for metric in knn_svm_config['knn_metrics']:
        for kernel in knn_svm_config['svm_kernels']:
            res_dir_name = 'knn_{}+svm_{}'.format(metric, kernel)
            res_dir = os.path.join(base_res_dir, res_dir_name)
            os.makedirs(res_dir, exist_ok=True)
            print('KNN ({}) + SVM ({})\n'.format(metric, kernel))

            knn_svm_metric_kernel_config = dict(knn_svm_config)
            del knn_svm_metric_kernel_config['knn_metrics']
            knn_svm_metric_kernel_config['knn_metric'] = metric
            del knn_svm_metric_kernel_config['svm_kernels']
            knn_svm_metric_kernel_config['svm_kernel'] = kernel

            # Print the configuration
            print_dict(knn_svm_metric_kernel_config)
            print()

            # Save method config file
            with open(os.path.join(res_dir, 'method_config.json'), 'w') as mcf:
                json.dump(knn_svm_metric_kernel_config, mcf, ensure_ascii=False, indent=4)

            # Open results file and run folds
            res_filename = os.path.join(res_dir, 'results.csv')
            with open(res_filename, 'w') as res_file:
                res_file.write('fold,test,expected_label,predicted_label\n')

                distance_metric = cosine if metric == 'cosine' else metric
                for i, (train, test) in enumerate(training_test_splits):
                    fold_i_res_dir = os.path.join(res_dir, 'fold_{}'.format(i))
                    os.makedirs(fold_i_res_dir)

                    knn_plus_svm = KNNPlusSVM(knn_svm_metric_kernel_config['k'], knn_metric=distance_metric,
                                              svm_kernel=kernel, res_dir=fold_i_res_dir)
                    run_fold(knn_plus_svm, dataset, train, test, i, fold_i_res_dir, res_file)

                    # Compress fold results and delete the original folder
                    with tarfile.open('{}.tar.gz'.format(fold_i_res_dir), 'w:gz') as tar:
                        tar.add(fold_i_res_dir, arcname=os.path.basename(fold_i_res_dir))
                    shutil.rmtree(fold_i_res_dir)

            print()
            process_exp_results(res_filename, print_dir_name=False)
            print('\n'+'-'*50+'\n')


def run_baseline_methods(baseline_config):
    print('Baseline Configuration\n')
    print_dict(baseline_config)

    # Create results directory
    res_dir = baseline_config['res_dir']
    os.makedirs(res_dir)
    with open(os.path.join(res_dir, 'baseline_config.json'), 'w') as json_config:
        json.dump(baseline_config, json_config, ensure_ascii=False, indent=4)

    # Load dataset
    dataset = pd.read_csv(baseline_config['dataset'], sep=',')
    shutil.copy2(baseline_config['dataset'], res_dir)

    # Copy class mapping file il present
    if 'class_mapping' in baseline_config:
        shutil.copy2(baseline_config['class_mapping'], res_dir)

    # K-fold cross-validation
    kf = StratifiedKFold(n_splits=baseline_config['k-folds'], shuffle=True,
                         random_state=baseline_config['k-fold_random_state'])
    columns = len(dataset.columns)
    training_test_splits = [
        (train.tolist(), test.tolist())
        for (train, test) in kf.split(dataset.iloc[:, :-1], dataset.iloc[:, columns - 1:columns])
    ]

    # Save the splits
    with open(os.path.join(res_dir, 'training_test_splits.json'), 'w') as tts:
        json.dump(training_test_splits, tts, ensure_ascii=False)

    if len(baseline_config['methods']) > 0:
        print('\n' + '-' * 50 + '\n')

    for method in baseline_config['methods']:
        if method == 'random_forest':
            run_random_forest(dataset, training_test_splits, baseline_config['methods'][method],
                              os.path.join(res_dir, 'random_forest'))
        elif method == 'svm':
            run_svm(dataset, training_test_splits, baseline_config['methods'][method], os.path.join(res_dir, 'svm'))
        elif method == 'knn':
            run_knn(dataset, training_test_splits, baseline_config['methods'][method], os.path.join(res_dir, 'knn'))
        elif method == 'knn+classifier':
            run_knn_plus_classifier(dataset, training_test_splits, baseline_config['methods'][method], res_dir)
        elif method == 'knn+svm':
            run_knn_plus_svm(dataset, training_test_splits, baseline_config['methods'][method], res_dir)
        else:
            print('Unknown method \'{}\''.format(method))
            print('\n' + '-' * 50 + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for running baseline methods')
    parser.add_argument('baseline_config_file', metavar='baseline_config_file', type=str, nargs='?', default=None,
                        help='file (.json) containing the configuration for the desired methods')
    args = parser.parse_args()

    with open(args.baseline_config_file) as bcf:
        baseline_config = json.load(bcf)
        preprocess_baseline_config(baseline_config, args.baseline_config_file)
        run_baseline_methods(baseline_config)