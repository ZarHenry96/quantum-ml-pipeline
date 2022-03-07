import argparse
import json
import os
import random
import re
import sys

from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_binary_classifier.classifier import run_classifier
from quantum_binary_classifier.get_classifier_job_res import retrieve_classifier_results

from quantum_knn.knn import run_knn
from quantum_knn.get_knn_job_res import retrieve_knn_results


# Colors for console prints
W = '\033[0m'   # white (normal)
Y = '\033[33m'  # yellow


def preprocess_pipeline_config(config, training_data_file, target_data_file, config_file_path, script_dir):
    if training_data_file is not None:
        config['training_data'] = os.path.abspath(training_data_file)
    elif not os.path.isabs(config['training_data']):
        config['training_data'] = \
            os.path.abspath(os.path.join(os.path.dirname(config_file_path), config['training_data']))

    if target_data_file is not None:
        config['target_data'] = os.path.abspath(target_data_file)
    elif not os.path.isabs(config['target_data']):
        config['target_data'] = \
            os.path.abspath(os.path.join(os.path.dirname(config_file_path), config['target_data']))

    if 'res_root_dir' in config and config['res_root_dir'] != 'default':
        if not os.path.isabs(config['res_root_dir']):
            config['res_root_dir'] = os.path.abspath(config['res_root_dir'])
    else:
        config['res_root_dir'] = os.path.abspath(os.path.join(script_dir, 'results'))


def knn_step(config, knn_config, knn_res_dir, store_results, verbose):
    if store_results:
        os.makedirs(knn_res_dir)

    classical, statevector, local_simulation, online_simulation, backend_name, job_name, init_gate, shots, \
        automatic_online_wait = False, False, False, False, None, None, False, None, False

    if knn_config['exec_type'] == 'classical':
        classical = True
    elif knn_config['exec_type'] == 'statevector':
        statevector = True
    elif knn_config['exec_type'] == 'local_simulation':
        local_simulation, shots = True, knn_config['shots']
    elif knn_config['exec_type'] == 'online_simulation':
        online_simulation, backend_name, job_name, shots, automatic_online_wait = \
            True, knn_config['backend_name'], knn_config['job_name'], knn_config['shots'], \
            knn_config['automatic_online_wait']
    elif knn_config['exec_type'] == 'quantum':
        backend_name, job_name, shots, automatic_online_wait = \
            knn_config['backend_name'], knn_config['job_name'], knn_config['shots'], \
            knn_config['automatic_online_wait']
    else:
        print('Unknown exec_type \'{}\''.format(knn_config['exec_type']), file=sys.stderr)
        exit(0)

    normalized_knn_df, normalized_ti_df, normalized_knn_df_file_path, normalized_ti_df_file_path = \
        run_knn(config['training_data'], config['target_data'], config['attribute_normalization'], knn_config['k'],
                classical, statevector, local_simulation, online_simulation, backend_name, job_name, init_gate, shots,
                knn_res_dir, store_results, verbose, automatic_online_wait=automatic_online_wait)
    
    if normalized_knn_df_file_path is None and normalized_ti_df_file_path is None:
        random_id = random.randint(0, 1000000)

        normalized_knn_df_file_path = os.path.join(
            '/tmp',
            'normalized_knn_{}_{}.csv'.format(knn_config['exec_type'], random_id)
        )
        normalized_knn_df.to_csv(normalized_knn_df_file_path, index=False)

        normalized_ti_df_file_path = os.path.join(
            '/tmp',
            'normalized_ti_{}_{}.csv'.format(knn_config['exec_type'], random_id)
        )
        normalized_ti_df.to_csv(normalized_ti_df_file_path, index=False)
    
    return normalized_knn_df, normalized_ti_df, normalized_knn_df_file_path, normalized_ti_df_file_path


def classifier_step(config, classifier_config, classifier_df_file_path, classifier_ti_file_path, classifier_res_dir,
                    store_results, verbose, already_normalized):
    if store_results:
        os.makedirs(classifier_res_dir)

    classical, statevector, local_simulation, online_simulation, backend_name, job_name, init_gate, shots, \
        automatic_online_wait = False, False, False, False, None, None, False, None, False

    if classifier_config['exec_type'] == 'classical':
        classical = True
    elif classifier_config['exec_type'] == 'statevector':
        statevector = True
    elif classifier_config['exec_type'] == 'local_simulation':
        local_simulation, shots = True, classifier_config['shots']
    elif classifier_config['exec_type'] == 'online_simulation':
        online_simulation, backend_name, job_name, shots, automatic_online_wait = \
            True, classifier_config['backend_name'], classifier_config['job_name'], \
            classifier_config['shots'], classifier_config['automatic_online_wait']
    elif classifier_config['exec_type'] == 'quantum':
        backend_name, job_name, shots, automatic_online_wait = \
            classifier_config['backend_name'], classifier_config['job_name'], \
            classifier_config['shots'], classifier_config['automatic_online_wait']
    else:
        print('Unknown exec_type \'{}\''.format(classifier_config['exec_type']), file=sys.stderr)
        exit(0)

    label = run_classifier(classifier_df_file_path, classifier_ti_file_path, config['attribute_normalization'],
                           classical, statevector, local_simulation, online_simulation, backend_name, job_name,
                           init_gate, shots, classifier_res_dir, store_results, verbose,
                           already_normalized=already_normalized, automatic_online_wait=automatic_online_wait)
    return label


def run_pipeline(config):
    store_results = config['store']
    verbose = config['verbose']
    
    # Algorithms configurations
    knn_config = config['knn']
    classifier_config = config['classifier']
    
    # Create results directory (if required)
    res_dir = os.path.join(config['res_root_dir'], '{}_knn_{}_classifier_{}'.format(
        knn_config['exec_type'],
        classifier_config['exec_type'],
        datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    ))
    if store_results:
        os.makedirs(res_dir)
        with open(os.path.join(res_dir, 'config.json'), 'w') as json_config:
            json.dump(config, json_config, ensure_ascii=False, indent=4)
    
    # Run KNN
    if knn_config['exec_type'] != 'none':
        if verbose:
            print(f'{Y}KNN step ({knn_config["exec_type"]}){W}\n')
        knn_res_dir = os.path.join(res_dir, '1_{}_knn_{}'.format(
            knn_config['exec_type'],
            datetime.now().strftime('%d-%m-%Y_%H-%M-%S'))
        )
        knn_df, ti_df, knn_df_file_path, ti_df_file_path = \
            knn_step(config, knn_config, knn_res_dir, store_results, verbose)
        already_normalized = True
    else:
        knn_df_file_path, ti_df_file_path = config['training_data'], config['target_data']
        already_normalized = False
    
    # Run classifier
    if verbose:
        print(f'\n\n{Y}Classifier step ({classifier_config["exec_type"]}){W}\n')
    classifier_res_dir = os.path.join(res_dir, '2_{}_classifier_{}'.format(
        classifier_config['exec_type'],
        datetime.now().strftime('%d-%m-%Y_%H-%M-%S'))
    )
    label = classifier_step(config, classifier_config, knn_df_file_path, ti_df_file_path, classifier_res_dir,
                            store_results, verbose, already_normalized)

    # Delete potential temporary file:
    if not store_results and knn_config['exec_type'] != 'none':
        os.remove(knn_df_file_path)
        os.remove(ti_df_file_path)

    return label


def resume_execution(res_dir):
    label = None

    # Read json config file
    config_file_path = os.path.join(res_dir, 'config.json')
    with open(config_file_path) as cf:
        config = json.load(cf)
    verbose = config['verbose']

    # Check if execution must be resumed from knn
    done = False
    if config['knn']['exec_type'] == 'online_simulation' or config['knn']['exec_type'] == 'quantum':
        knn_dirname = [
            x for x in os.listdir(res_dir)
            if re.match('1_{}_knn_.*'.format(config['knn']['exec_type']), x)
        ][0]
        knn_out_dir = os.path.join(res_dir, knn_dirname, 'output')

        # Retrieve the results if the output directory is empty
        if len(os.listdir(knn_out_dir)) == 0:
            normalized_knn_df, normalized_ti_df, normalized_knn_df_file_path, normalized_ti_df_file_path = \
                retrieve_knn_results(os.path.join(res_dir, knn_dirname), verbose=verbose)

            # Run the classifier if the job is ended
            if normalized_knn_df is not None and normalized_ti_df is not None and \
                    normalized_knn_df_file_path is not None and normalized_ti_df_file_path is not None:
                if verbose:
                    print(f'\n\n{Y}Classifier step ({config["classifier"]["exec_type"]}){W}\n')
                classifier_res_dir = os.path.join(res_dir, '2_{}_classifier_{}'.format(
                                                                config['classifier']['exec_type'],
                                                                datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
                                                            )
                )
                already_normalized = True
                label = classifier_step(config, config['classifier'], normalized_knn_df_file_path,
                                        normalized_ti_df_file_path, classifier_res_dir, config['store'],
                                        verbose, already_normalized)

            # End the execution
            done = True

    # Check if the execution must be resumed from classifier
    if not done and \
            (config['classifier']['exec_type'] == 'online_simulation' or
             config['classifier']['exec_type'] == 'quantum'):
        classifier_dirname = [
            x for x in os.listdir(res_dir)
            if re.match('2_{}_classifier_.*'.format(config['classifier']['exec_type']), x)
        ][0]
        classifier_out_dir = os.path.join(res_dir, classifier_dirname, 'output')

        # Retrieve the results if the output directory is empty
        if len(os.listdir(classifier_out_dir)) == 0:
            label = retrieve_classifier_results(os.path.join(res_dir, classifier_dirname), verbose=verbose)
        else:
            label_filename = [
                x for x in os.listdir(classifier_out_dir)
                if re.match('output_label_{}_.*'.format(config['classifier']['exec_type']), x)
            ][0]
            with open(label_filename) as lf:
                label = int(lf.readline().strip().split(' ')[4])

    return label


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for executing a KNN-classifier pipeline')
    parser.add_argument('config_file', metavar='config_file', type=str, nargs='?', default=None,
                        help='file (.json) containing the configuration parameters for KNN and classifier')
    parser.add_argument('--training', metavar='training_data_file', type=str, nargs='?', default=None,
                        help='file (.csv) containing the training data: this overrides the training_data parameter '
                             'in the configuration file (if present)')
    parser.add_argument('--target', metavar='target_instance_file', type=str, nargs='?', default=None,
                        help='file (.csv) containing the target instance: this overrides the target_data parameter '
                             'in the configuration file (if present)')
    parser.add_argument('--resume-exec-from-res-dir', metavar='resume_exec_from_res_dir', type=str, nargs='?',
                        default=None, help='results directory from which to resume the execution')
    args = parser.parse_args()

    script_dir = os.path.join('./', os.path.dirname(sys.argv[0]))

    if args.resume_exec_from_res_dir is not None:
        resume_execution(args.resume_exec_from_res_dir)
    else:
        with open(args.config_file) as cf:
            config = json.load(cf)
            preprocess_pipeline_config(config, args.training, args.target, args.config_file, script_dir)
            run_pipeline(config)
