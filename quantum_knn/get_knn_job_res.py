from qiskit import IBMQ
from qiskit.providers.jobstatus import JOB_FINAL_STATES
from qiskit.visualization import plot_histogram

import argparse
import numpy as np
import os
import pandas as pd
import sys

from datetime import datetime, timezone
from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quantum_knn.knn import process_counts, print_knn_qcircuit_results, save_knn_qcircuit_log


def retrieve_knn_results(res_dir, verbose=True):
    # Read the execution info from the knn_job_info.txt file
    job_info_file = os.path.join(res_dir, 'knn_job_info.txt')
    with open(job_info_file) as job_info_file:
        job_id = job_info_file.readline().strip()
        exec_modality = job_info_file.readline().strip()
        backend_name = job_info_file.readline().strip()
        init_modality = job_info_file.readline().strip()
        index_qubits = int(job_info_file.readline().strip())
        training_data_file = os.path.join(res_dir, 'input', job_info_file.readline().strip())
        normalized_training_data_file = os.path.join(res_dir, 'input', job_info_file.readline().strip())
        k = int(job_info_file.readline().strip())
        normalized_unclassified_instance_file = os.path.join(res_dir, 'input', job_info_file.readline().strip())

    # Load the training data
    original_t_df = pd.read_csv(training_data_file, sep=',')
    normalized_t_df = pd.read_csv(normalized_training_data_file, sep=',')

    # Load the normalized unclassified instance
    normalized_ui_df = pd.read_csv(normalized_unclassified_instance_file, sep=',')

    # Create output folder (if not present)
    output_dir = os.path.join(res_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    # Get the IBMQ backend
    if IBMQ.active_account() is not None:
        IBMQ.disable_account()
    provider = IBMQ.load_account()
    backend = provider.get_backend(backend_name)

    # Get the status of the Job
    retrieved_job = backend.retrieve_job(job_id)
    job_status = retrieved_job.status()
    if job_status not in JOB_FINAL_STATES:
        current_time = datetime.now(timezone.utc)

        queue_info = retrieved_job.queue_info()
        if queue_info is None:
            position, run_time = None, None
        else:
            position = queue_info.position
            run_time = str(queue_info.estimated_complete_time - current_time).split(".")[0] \
                if current_time < queue_info.estimated_complete_time \
                else 'new estimation in progress'

        if verbose:
            print(f'Job {job_id} \n\t status: {job_status.name},'
                  f'\n\t est. queue position: {position}'
                  f'\n\t est. run time: {run_time}')

        return None, None, None, None

    # Grab results from the Job if it is completed
    result = retrieved_job.result()
    counts = result.get_counts()
    sorted_counts = {i: counts[i] for i in sorted(counts)}
    if verbose:
        print('Results\nTotal output counts are: {}'.format(sorted_counts))

    # Save counts to txt file and as histogram
    counts_filename = 'knn_counts_{}_{}'.format(exec_modality, init_modality)
    # Txt file
    with open(os.path.join(output_dir, counts_filename + '.txt'), 'w') as counts_file:
        counts_file.write(f'{counts}')
    # Histogram
    plot_histogram(counts)
    plt.savefig(os.path.join(output_dir, counts_filename + '.pdf'))
    plt.close()

    # Process counts
    p0, p1, index_cond_p, index_cond_p_diff_sorted = \
        process_counts(counts, index_qubits, len(original_t_df))

    # Get the k nearest neighbours
    nearest_neighbours_df = original_t_df.iloc[list(index_cond_p_diff_sorted.keys())[0: k], :]
    normalized_nearest_neighbours_df = normalized_t_df.iloc[list(index_cond_p_diff_sorted.keys())[0: k], :]

    # Display and store the results
    if verbose:
        print_knn_qcircuit_results(p0, p1, index_cond_p, index_cond_p_diff_sorted, k)

    log_filename = os.path.join(output_dir, 'knn_{}_{}_log.txt'.format(exec_modality, init_modality))
    save_knn_qcircuit_log(log_filename, p0, p1, index_cond_p, index_cond_p_diff_sorted, k)

    knn_filename = os.path.join(output_dir, 'knn_{}_{}.csv'.format(exec_modality, init_modality))
    nearest_neighbours_df.to_csv(knn_filename, index=False)

    normalized_knn_filename = os.path.join(output_dir, 'normalized_knn_{}_{}.csv'.format(exec_modality, init_modality))
    normalized_nearest_neighbours_df.to_csv(normalized_knn_filename, index=False)

    # Display the final outcome
    if verbose:
        print(f'\nThe normalized {k} nearest neighbours for the unclassified instance provided are:')
        print(normalized_nearest_neighbours_df)

    normalized_nearest_neighbours_df.reset_index(drop=True, inplace=True)
    return normalized_nearest_neighbours_df, normalized_ui_df, \
           normalized_knn_filename, normalized_unclassified_instance_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for retrieving the results of a quantum k-NN '
                                                 'Job executed on IBMQ.')
    parser.add_argument('job_info_folder', metavar='job_info_folder', type=str, nargs='?',
                        help='path to the results directory containing the knn_job_info.txt file '
                             'for the Job of interest')
    args = parser.parse_args()

    retrieve_knn_results(args.job_info_folder)
