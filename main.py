import argparse
import json
import multiprocessing
import os
import pandas as pd
import re
import shutil
import sys
import tarfile

from sklearn.model_selection import StratifiedKFold
from quantum_models_pipeline.pipeline import run_pipeline

from pre_post_processing.process_exp_results import process_exp_results


def preprocess_experiment_config(config, config_file_path):
    if not os.path.isabs(config['dataset']):
        config['dataset'] = os.path.abspath(os.path.join(os.path.dirname(config_file_path), config['dataset']))

    if 'class_mapping' in config and not os.path.isabs(config['class_mapping']):
        config['class_mapping'] = \
            os.path.abspath(os.path.join(os.path.dirname(config_file_path), config['class_mapping']))

    if not os.path.isabs(config['res_dir']):
        config['res_dir'] = os.path.abspath(os.path.join(os.path.dirname(config_file_path), config['res_dir']))


def print_dict(d, level=0, list_on_levels=False):
    for k, v in sorted(d.items()):
        if isinstance(v, dict):
            print('\t'*level + k+':')
            print_dict(v, level+1, list_on_levels)
        elif isinstance(v, list) and list_on_levels:
            print('\t' * level + '{}: ['.format(k))
            for element in v:
                print('\t' * (level+1) + '{}'.format(element))
            print('\t' * level + ']')
        else:
            print('\t'*level + '{}: {}'.format(k, v))


def mute():
    sys.stdout = open(os.devnull, 'w')


def run_test(test_config_file):
    mute()
    with open(test_config_file) as tcf:
        test_config = json.load(tcf)
        return run_pipeline(test_config)


def run_fold(config, dataset, train, test, i, fold_res_dir, res_file, pool):
    # Save the training and test set for the current fold
    training_data_file = os.path.join(fold_res_dir, 'training_data.csv')
    dataset.iloc[train].to_csv(training_data_file, index=False)

    full_test_instance_file = os.path.join(fold_res_dir, 'test_instances.csv')
    dataset.iloc[test].to_csv(full_test_instance_file, index=False)

    # Prepare folders, input and config files
    test_config_files = []
    expected_labels = []
    for j in range(0, len(test)):
        test_j_res_dir = os.path.join(fold_res_dir, 'test_{}'.format(j))
        os.makedirs(test_j_res_dir)

        test_instance_file = os.path.join(test_j_res_dir, 'test_instance.csv')
        dataset.iloc[[test[j]], :-1].to_csv(test_instance_file, index=False)

        instance_label_file = os.path.join(test_j_res_dir, 'instance_label.txt')
        with open(instance_label_file, 'w') as ilf:
            ilf.write(str(dataset.iloc[test[j], -1]))

        test_config = {
            'training_data': training_data_file,
            'target_data': test_instance_file,
            'attribute_normalization': config['attribute_normalization'],
            'res_root_dir': test_j_res_dir,
            'store': True,
            'knn': dict(config['knn']),
            'classifier': dict(config['classifier']),
            'verbose': False
        }
        if config['knn']['exec_type'] in ['online_simulation', 'quantum']:
            test_config['knn']['automatic_online_wait'] = True
        if config['classifier']['exec_type'] in ['online_simulation', 'quantum']:
            test_config['classifier']['automatic_online_wait'] = True

        test_config_file = os.path.join(test_j_res_dir, 'test_config.json')
        with open(test_config_file, 'w') as test_json_config:
            json.dump(test_config, test_json_config, ensure_ascii=False, indent=4)

        test_config_files.append(test_config_file)
        expected_labels.append(dataset.iloc[test[j], -1])

    # Parallel execution
    print('Fold {} ...'.format(i), end='')
    predicted_labels = pool.map(run_test, test_config_files)
    print('\tDone')

    # Save the fold results
    for j, (exp_label, pred_label) in enumerate(zip(expected_labels, predicted_labels)):
        res_file.write('{},{},{},{}\n'.format(i, j, exp_label, pred_label))

    # Compress fold results and delete the original folder
    with tarfile.open('{}.tar.gz'.format(fold_res_dir), 'w:gz') as tar:
        tar.add(fold_res_dir, arcname=os.path.basename(fold_res_dir))
    shutil.rmtree(fold_res_dir)


def run(config):
    # Show experiment configuration
    print('Experiment Configuration\n')
    print_dict(config)
    print('\n')

    # Create results directory
    res_dir = config['res_dir']
    os.makedirs(res_dir)
    with open(os.path.join(res_dir, 'exp_config.json'), 'w') as json_config:
        json.dump(config, json_config, ensure_ascii=False, indent=4)

    # Load dataset
    dataset = pd.read_csv(config['dataset'], sep=',')
    shutil.copy2(config['dataset'], res_dir)

    # Copy class mapping file il present
    if 'class_mapping' in config:
        shutil.copy2(config['class_mapping'], res_dir)

    # Create results file
    res_filename = os.path.join(res_dir, 'results.csv')
    with open(res_filename, 'w') as res_file:
        res_file.write('fold,test,expected_label,predicted_label\n')

        # K-fold cross-validation
        kf = StratifiedKFold(n_splits=config['k-folds'], shuffle=True, random_state=config['k-fold_random_state'])
        columns = len(dataset.columns)
        training_test_splits = [
            (train.tolist(), test.tolist())
            for (train, test) in kf.split(dataset.iloc[:, :-1], dataset.iloc[:, columns-1:columns])
        ]

        # Save the splits for a potential resume of execution
        with open(os.path.join(res_dir, 'training_test_splits.json'), 'w') as tts:
            json.dump(training_test_splits, tts, ensure_ascii=False)

        with multiprocessing.Pool(processes=config['num_processes']) as pool:
            # Iterate over folds
            for i, (train, test) in enumerate(training_test_splits):
                fold_i_res_dir = os.path.join(res_dir, 'fold_{}'.format(i))
                os.makedirs(fold_i_res_dir)

                run_fold(config, dataset, train, test, i, fold_i_res_dir, res_file, pool)

    print('\n')
    process_exp_results(res_filename, print_dir_name=False)


def parse_test_dir(config, j, test_dir, fold_results, missing_tests,
                   missing_tests_config_files, missing_tests_expected_labels):
    test_config_file = os.path.join(test_dir, 'test_config.json')

    # Get the expected label
    with open(os.path.join(test_dir, 'instance_label.txt')) as ilf:
        expected_label = int(ilf.readline().strip())

    # Check if the pipeline directory exists
    pipeline_dir = [
        d for d in os.listdir(test_dir)
        if re.match('{}_knn_{}_classifier_.*'.format(config['knn']['exec_type'], config['classifier']['exec_type']), d)
    ]

    if len(pipeline_dir) > 0:
        assert len(pipeline_dir) == 1
        pipeline_dir = os.path.join(test_dir, pipeline_dir[0])

        # Check if the classifier results directory exists
        classifier_res_dir = [
            d for d in os.listdir(pipeline_dir)
            if re.match('2_{}_classifier_.*'.format(config['classifier']['exec_type']), d)
        ]

        if len(classifier_res_dir) > 0:
            assert len(classifier_res_dir) == 1
            classifier_res_dir = os.path.join(pipeline_dir, classifier_res_dir[0])

            # Check if the output file exists
            output_file = [
                f for f in os.listdir(os.path.join(classifier_res_dir, 'output'))
                if re.match('output_label_{}_.*'.format(config['classifier']['exec_type']), f)
            ]

            if len(output_file) > 0:
                assert len(output_file) == 1
                output_file = os.path.join(classifier_res_dir, 'output', output_file[0])

                # Get the predicted label
                with open(output_file) as of:
                    predicted_label = int(of.readline().strip().split()[4])
                    fold_results[j] = (expected_label, predicted_label)
            else:
                # Delete the pipeline results directory and re-execute the test
                shutil.rmtree(pipeline_dir)
                missing_tests.append(j)
                missing_tests_config_files.append(test_config_file)
                missing_tests_expected_labels.append(expected_label)
        else:
            # Delete the pipeline results directory and re-execute the test
            shutil.rmtree(pipeline_dir)
            missing_tests.append(j)
            missing_tests_config_files.append(test_config_file)
            missing_tests_expected_labels.append(expected_label)
    else:
        missing_tests.append(j)
        missing_tests_config_files.append(test_config_file)
        missing_tests_expected_labels.append(expected_label)


def resume_execution(res_dir):
    # Read json config file
    config_file_path = os.path.join(res_dir, 'exp_config.json')
    with open(config_file_path) as cf:
        config = json.load(cf)

    # Load dataset
    dataset = pd.read_csv(os.path.join(res_dir, os.path.basename(config['dataset'])), sep=',')

    # Load results file
    results_filename = os.path.join(res_dir, 'results.csv')
    results_df = pd.read_csv(results_filename, sep=',')

    # Load k-fold splitting
    with open(os.path.join(res_dir, 'training_test_splits.json')) as ttf:
        training_test_splits = json.load(ttf)

    # Find last completed fold
    completed_folds = [int(re.split('_|\.', f)[1]) for f in os.listdir(res_dir) if re.match('fold_[0-9]+\.tar\.gz', f)]
    last_completed_fold = max(completed_folds)
    assert results_df.iloc[-1, 0] == last_completed_fold

    next_fold = last_completed_fold + 1
    if next_fold < config['k-folds']:
        print('Resume execution from fold {}'.format(next_fold))

    # Retrieve execution from next fold
    next_fold_res_dir = os.path.join(res_dir, 'fold_{}'.format(next_fold))
    if os.path.exists(next_fold_res_dir):
        next_fold_tests = len(training_test_splits[next_fold][1])
        if os.path.exists(os.path.join(next_fold_res_dir, 'test_{}'.format(next_fold_tests-1), 'test_config.json')):
            # The execution was interrupted after the preparation of the tests tree structure
            fold_results = {}
            missing_tests, missing_tests_config_files, missing_tests_expected_labels, missing_tests_predicted_labels = \
                [], [], [], []

            # Classify tests in completed and missing
            for j in range(0, next_fold_tests):
                test_j_res_dir = os.path.join(next_fold_res_dir, 'test_{}'.format(j))
                parse_test_dir(config, j, test_j_res_dir, fold_results, missing_tests,
                               missing_tests_config_files, missing_tests_expected_labels)

            # Run the missing tests:
            if len(missing_tests) > 0:
                with multiprocessing.Pool(processes=config['num_processes']) as pool:
                    print('Running missing tests: {} ...'.format(', '.join([str(x) for x in missing_tests])), end='')
                    missing_tests_predicted_labels = pool.map(run_test, missing_tests_config_files)
                    print('Done\n')

            # Get the missing results
            for j, exp_label, pred_label in zip(missing_tests, missing_tests_expected_labels,
                                                 missing_tests_predicted_labels):
                fold_results[j] = (exp_label, pred_label)

            # Sort the results by index and save them
            fold_results = {key: fold_results[key] for key in sorted(fold_results.keys())}
            with open(results_filename, 'a') as res_file:
                for j, (exp_label, pred_label) in fold_results.items():
                    res_file.write('{},{},{},{}\n'.format(next_fold, j, exp_label, pred_label))

            # Compress fold results and delete the original folder
            with tarfile.open('{}.tar.gz'.format(next_fold_res_dir), 'w:gz') as tar:
                tar.add(next_fold_res_dir, arcname=os.path.basename(next_fold_res_dir))
            shutil.rmtree(next_fold_res_dir)

            next_fold += 1
        else:
            shutil.rmtree(next_fold_res_dir)

    # Resume normal execution
    with open(results_filename, 'a') as res_file:
        with multiprocessing.Pool(processes=config['num_processes']) as pool:
            for i in range(next_fold, config['k-folds']):
                fold_i_res_dir = os.path.join(res_dir, 'fold_{}'.format(i))
                os.makedirs(fold_i_res_dir)

                run_fold(config, dataset, training_test_splits[i][0], training_test_splits[i][1],
                         i, fold_i_res_dir, res_file, pool)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for running experiments on the KNN-classifier pipeline')
    parser.add_argument('config_file', metavar='config_file', type=str, nargs='?', default=None,
                        help='file (.json) containing the configuration for the experiment')
    parser.add_argument('--resume-exec-from-res-dir', metavar='resume_exec_from_res_dir', type=str, nargs='?',
                        default=None, help='results directory from which to resume the execution')
    args = parser.parse_args()

    if args.resume_exec_from_res_dir is not None:
        resume_execution(args.resume_exec_from_res_dir)
    else:
        with open(args.config_file) as cf:
            config = json.load(cf)
            preprocess_experiment_config(config, args.config_file)
            run(config)
