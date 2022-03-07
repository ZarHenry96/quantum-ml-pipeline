from qiskit import IBMQ
from qiskit.providers.jobstatus import JOB_FINAL_STATES
from qiskit.visualization import plot_histogram

import argparse
import os

from datetime import datetime, timezone
from matplotlib import pyplot as plt


def retrieve_classifier_results(res_dir, old_mode=False, verbose=True):
    # Read the execution info from the classifier_job_info.txt file
    job_info_filename = 'job_info.txt' if old_mode else 'classifier_job_info.txt'
    job_info_file = os.path.join(res_dir, job_info_filename)
    with open(job_info_file) as job_info_file:
        job_id = job_info_file.readline().strip()
        exec_modality = job_info_file.readline().strip()
        backend_name = job_info_file.readline().strip()
        init_modality = job_info_file.readline().strip()

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
                  f'\n\t est. queue position: {position},'
                  f'\n\t est. run time: {run_time}')
        return None

    # Grab results from the Job if it is completed
    result = retrieved_job.result()
    counts = result.get_counts()
    sorted_counts = {i: counts[i] for i in sorted(counts)}
    if verbose:
        print('Results\nTotal output counts are: {}'.format(sorted_counts))

    # Save counts to txt file and as histogram
    counts_filename = 'classifier_counts_{}_{}'.format(exec_modality, init_modality)
    # Txt file
    with open(os.path.join(output_dir, counts_filename + '.txt'), 'w') as counts_file:
        counts_file.write(f'{counts}')
    # Histogram
    plot_histogram(counts)
    plt.savefig(os.path.join(output_dir, counts_filename + '.pdf'))
    plt.close()

    # Compute the probability of measuring 0 and 1
    p0 = counts['0'] / (counts['0'] + counts['1'])
    p1 = counts['1'] / (counts['0'] + counts['1'])

    # Display the probabilities
    if verbose:
        print(f'\nProbability of measuring 0: {p0}')
        print(f'Probability of measuring 1: {p1}')

    # Predict and display the label
    ui_label = -1 if p1 > 0.25 else 1
    if verbose:
        print(f'\nLabel of unclassified instance: {ui_label}\t(value={1 - 4 * p1})')

    label_filename = 'output_label_{}_{}.txt'.format(exec_modality, init_modality)
    with open(os.path.join(output_dir, label_filename), 'w') as label_file:
        label_file.write(f'Label of unclassified instance: {ui_label}\t(value={1-4*p1})')

    return ui_label


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for retrieving the results of a quantum binary classifier '
                                                 'Job executed on IBMQ.')
    parser.add_argument('job_info_folder', metavar='job_info_folder', type=str, nargs='?',
                        help='path to the results directory containing the classifier_job_info.txt file '
                             'for the Job of interest')
    parser.add_argument('--old-mode', dest='old_mode', action='store_const',
                        const=True, default=False, help='retro-compatibility (job filename without model prefix)')
    args = parser.parse_args()

    retrieve_classifier_results(args.job_info_folder, args.old_mode)
