from qiskit import (
    ClassicalRegister,
    QuantumRegister,
    QuantumCircuit,
    execute,
)
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram

import argparse
import math
import numpy as np
import os
import sys

from datetime import datetime
from matplotlib import pyplot as plt
from scipy.spatial.distance import cosine
from sklearn.neighbors import NearestNeighbors
from signal import signal, SIGINT

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_binary_classifier.classifier import load_data, select_backend, save_job_info, \
    wait_for_job_completion, save_output_statevector


# Colors for console prints
W  = '\033[0m'  # white (normal)
R  = '\033[31m' # red


def classical_knn(t_df, ui_df, k, original_t_df, save_to_file, res_dir, expectation=False, verbose=True):
    output_dir = os.path.join(res_dir, 'classical_expectation' if expectation else 'output')
    if save_to_file:
        os.makedirs(output_dir, exist_ok=True)

    features_number = len(ui_df.columns)

    # Prepare data for KNN
    training_instances = np.array(t_df.iloc[:, :features_number])
    target_instance = np.array(ui_df.iloc[:, :features_number])

    # Fit the model to the training data
    knn_model = NearestNeighbors(n_neighbors=k, metric=cosine, algorithm='brute')
    knn_model.fit(training_instances)

    # Determine the nearest neighbours
    nearest_neighbours = knn_model.kneighbors(target_instance)
    nearest_neighbours_df = original_t_df.iloc[nearest_neighbours[1][0], :]
    normalized_nearest_neighbours_df = t_df.iloc[nearest_neighbours[1][0], :]

    # Show results
    verb_to_print = 'expected' if expectation else 'predicted'
    if verbose:
        print(f'\nClassically {verb_to_print} {k} nearest neighbours for the unclassified instance provided:')
        for index, distance in zip(nearest_neighbours[1][0], nearest_neighbours[0][0]):
            element = np.array2string(training_instances[index], separator=', ')
            print(f'\tdistance: {distance}, index: {index}, element: {element}')

    # Save results (if required)
    normalized_knn_filename = None
    if save_to_file:
        prefix = 'expected_' if expectation else ''
        postfix = '(classical)' if expectation else 'classical'
        with open(os.path.join(output_dir, f'{prefix}knn_{postfix}_log.txt'), 'w') \
                as classical_knn_log:
            classical_knn_log.write(f'Classically {verb_to_print} {k} nearest neighbours for '
                                    f'the unclassified instance provided:')
            for index, distance in zip(nearest_neighbours[1][0], nearest_neighbours[0][0]):
                element = np.array2string(training_instances[index], separator=', ')
                classical_knn_log.write(f'\n\tdistance: {distance}, index: {index}, '
                                        f'element: {element}')

        knn_filename = os.path.join(output_dir, f'{prefix}knn_{postfix}.csv')
        nearest_neighbours_df.to_csv(knn_filename, index=False)

        normalized_knn_filename = os.path.join(output_dir, f'{prefix}normalized_knn_{postfix}.csv')
        normalized_nearest_neighbours_df.to_csv(normalized_knn_filename, index=False)

    normalized_nearest_neighbours_df.reset_index(drop=True, inplace=True)
    return normalized_nearest_neighbours_df, normalized_knn_filename


def joint_initialization(circuit, qr, index_qubits, features_qubits, t_df, ui_df, N, d):
    # Initialize jointly the index register and the training data features register
    init_qubits = index_qubits + features_qubits
    itd_amplitudes = np.zeros(2 ** init_qubits)
    amplitude_base_value = 1.0 / math.sqrt(N)

    # Training data amplitudes
    for instance_index, row in t_df.iterrows():
        for feature_indx, feature_amplitude in enumerate(row[0:-1]):
            index = instance_index + (2 ** index_qubits) * feature_indx
            itd_amplitudes[index] = amplitude_base_value * feature_amplitude

    # Set all index_register+td_features_register amplitudes
    circuit.initialize(itd_amplitudes, qr[1: 1+init_qubits])

    # Initialize the unclassified instance features register
    ui_amplitudes = np.zeros(2 ** features_qubits)
    for feature_indx, feature_amplitude in enumerate(ui_df.iloc[0, 0:d]):
        ui_amplitudes[feature_indx] = feature_amplitude

    # Set all ui_features_register amplitudes
    circuit.initialize(ui_amplitudes, qr[1+index_qubits+features_qubits: 1+index_qubits+2*features_qubits])

    # Add swap test gates
    circuit.h(qr[0])
    for i in range(0, features_qubits):
        circuit.cswap(qr[0], qr[1+index_qubits+i], qr[1+index_qubits+features_qubits+i])
    circuit.h(qr[0])


def process_output_statevector(output_statevector, qubits_num, index_qubits, N):
    # Compute for any index value i the probability P(i|0)-P(i|1), which is proportional to the similarity
    # between the considered training instance and the target one
    p0, p1, index_cond_p = 0.0, 0.0, {}
    for i in range(0, (2 ** index_qubits)):
        bin_val = ('{0:0' + str(index_qubits) + 'b}').format(i)
        index_cond_p[bin_val] = {'0': 0.0, '1': 0.0}

    for state, state_amplitude in enumerate(output_statevector):
        if state % 2 == 0:
            p0 += (np.abs(state_amplitude) ** 2)
        else:
            p1 += (np.abs(state_amplitude) ** 2)

        bin_state = ('{0:0' + str(qubits_num) + 'b}').format(state)
        index_state = bin_state[qubits_num - (index_qubits + 1):-1]
        ancillary_state = bin_state[qubits_num - 1]
        index_cond_p[index_state][ancillary_state] += (np.abs(state_amplitude) ** 2)

    for index_state in index_cond_p.keys():
        index_cond_p[index_state]['0'] /= (p0 if p0 != 0 else 1)
        index_cond_p[index_state]['1'] /= (p1 if p1 != 0 else 1)

    index_cond_p_diff = {}
    for index_state in index_cond_p.keys():
        index_state_int = int(index_state, 2)
        if (index_cond_p[index_state]['0'] != 0 or index_cond_p[index_state]['1'] != 0) and index_state_int < N:
            index_cond_p_diff[index_state_int] = \
                index_cond_p[index_state]['0'] - index_cond_p[index_state]['1']
    index_cond_p_diff_sorted = {
        d_key: d_val
        for d_key, d_val in sorted(index_cond_p_diff.items(), key=lambda item: item[1], reverse=True)
    }

    return p0, p1, index_cond_p, index_cond_p_diff_sorted


def process_counts(counts, index_qubits, N):
    # Prepare data structures
    counts_0, counts_1, index_cond_p = 0, 0, {}
    for i in range(0, (2 ** index_qubits)):
        bin_val = ('{0:0' + str(index_qubits) + 'b}').format(i)
        index_cond_p[bin_val] = {'0': 0, '1': 0}

    # Process counts
    for state, state_counts in counts.items():
        index_state = state[0: -1]
        ancillary_state = state[index_qubits]
        index_cond_p[index_state][ancillary_state] += state_counts

        if ancillary_state == '0':
            counts_0 += state_counts
        else:
            counts_1 += state_counts

    # Estimate the probability values of interest as relative frequencies
    p0 = counts_0 / (counts_0 + counts_1)
    p1 = counts_1 / (counts_0 + counts_1)
    for index_state in index_cond_p.keys():
        index_cond_p[index_state]['0'] /= (counts_0 if counts_0 != 0 else 1)
        index_cond_p[index_state]['1'] /= (counts_1 if counts_1 != 0 else 1)

    # Compute the conditional probability differences
    index_cond_p_diff = {}
    for index_state in index_cond_p.keys():
        index_state_int = int(index_state, 2)
        if index_state_int < N:
            index_cond_p_diff[index_state_int] = \
                index_cond_p[index_state]['0'] - index_cond_p[index_state]['1']
    index_cond_p_diff_sorted = {
        d_key: d_val
        for d_key, d_val in sorted(index_cond_p_diff.items(), key=lambda item: item[1], reverse=True)
    }

    return p0, p1, index_cond_p, index_cond_p_diff_sorted


def print_knn_qcircuit_results(p0, p1, index_cond_p, index_cond_p_diff_sorted, k):
    print(f'\nP(ancillary_qubit_state):')
    print(f'\tP(0) = {p0}')
    print(f'\tP(1) = {p1}')

    print(f'\nP(index_state|ancillary_qubit_state):')
    for index_state, cond_p in index_cond_p.items():
        dec_state = int(index_state, 2)
        print(f'\tIndex state {dec_state} (binary: {index_state})')
        print(f'\t\tP({dec_state}|0) = {cond_p["0"]}')
        print(f'\t\tP({dec_state}|1) = {cond_p["1"]}')

    print(f'\nSorted \'P(index_state|0_ancillary)-P(index_state|1_ancillary)\' (without nonexistent index_states):')
    for i, (key, value) in enumerate(index_cond_p_diff_sorted.items()):
        if i < k:
            print(f'\ti={key}: {value}')
        else:
            print(f'{R}\ti={key}: {value}{W}')


def save_knn_qcircuit_log(filename, p0, p1, index_cond_p, index_cond_p_diff_sorted, k):
    with open(filename, 'w') as knn_qcircuit_log:
        knn_qcircuit_log.write('P(ancillary_qubit_state):\n')
        knn_qcircuit_log.write(f'\tP(0) = {p0}\n')
        knn_qcircuit_log.write(f'\tP(1) = {p1}\n')

        knn_qcircuit_log.write(f'\nP(index_state|ancillary_qubit_state):\n')
        for index_state, cond_p in index_cond_p.items():
            dec_state = int(index_state, 2)
            knn_qcircuit_log.write(f'\tIndex state {dec_state} (binary: {index_state})\n')
            knn_qcircuit_log.write(f'\t\tP({dec_state}|0) = {cond_p["0"]}\n')
            knn_qcircuit_log.write(f'\t\tP({dec_state}|1) = {cond_p["1"]}\n')

        knn_qcircuit_log.write('\nSorted \'P(index_state|0_ancillary)-P(index_state|1_ancillary)\' '
                               '(without nonexistent index_states):\n')
        for i, (key, value) in enumerate(index_cond_p_diff_sorted.items()):
            knn_qcircuit_log.write(f'\ti={key}: {value}\n')

        knn_qcircuit_log.write(f'\nThe first {k} instances in terms of '
                               f'\'P(index_state|0_ancillary)-P(index_state|1_ancillary)\' '
                               f'are selected as nearest neighbours')


def run_knn(training_data_file, unclassified_instance_file, attribute_normalization, k, classical, statevector,
            local_simulation, online_simulation, backend_name, job_name, init_gate, shots, res_dir, store_results,
            verbose, automatic_online_wait=False):
    input_dir = os.path.join(res_dir, 'input')
    output_dir = os.path.join(res_dir, 'output')
    if store_results:
        os.makedirs(res_dir, exist_ok=True)
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

    # Get data from input files
    instance_normalization = True
    t_df, ui_df, normalized_data_files = load_data(training_data_file, unclassified_instance_file,
                                                   attribute_normalization, instance_normalization,
                                                   store_results, input_dir, show=verbose)
    N, d = len(t_df), len(t_df.columns) - 1

    # Get original t_df
    original_t_df, _, _ = load_data(training_data_file, unclassified_instance_file, False, False,
                                    False, None, show=False)

    # Save k value
    if store_results:
        with open(os.path.join(res_dir, 'k_value.txt'), 'w') as k_value_file:
            k_value_file.write(f'{k}')

    # Execute only the classical KNN
    if classical:
        normalized_nearest_neighbours_df, normalized_knn_filename = \
            classical_knn(t_df, ui_df, k, original_t_df, store_results, res_dir, verbose=verbose)
        return normalized_nearest_neighbours_df, ui_df, normalized_knn_filename, normalized_data_files[1]

    # Select the backend for the execution
    backend = select_backend(statevector, local_simulation, online_simulation, backend_name)

    # Compute the expected label value (classical computation)
    classical_knn(t_df, ui_df, k, original_t_df, store_results, res_dir, expectation=True, verbose=verbose)

    # Compute circuit size
    swap_circuit_qubits = 1
    index_qubits = math.ceil(math.log2(N))
    features_qubits = math.ceil(math.log2(d))
    qubits_num = swap_circuit_qubits + index_qubits + 2*features_qubits
    c_bits_num = swap_circuit_qubits + index_qubits

    # Create a Quantum Circuit acting on the q register
    qr = QuantumRegister(qubits_num, 'q')
    cr = ClassicalRegister(c_bits_num, 'c')
    circuit = QuantumCircuit(qr, cr)

    # Build the circuit using the selected method
    if not init_gate:
        init_modality = 'joint_init'
        joint_initialization(circuit, qr, index_qubits, features_qubits, t_df, ui_df, N, d)
    else:
        init_modality = 'init_gate'
        print('Not implemented yet', file=sys.stderr)
        exit(0)

    # Measure qubits for swap test
    if statevector:
        circuit.save_statevector()
    else:
        circuit.measure(qr[0], cr[0])
        circuit.barrier(qr[0: c_bits_num])
        circuit.measure(qr[1: c_bits_num], cr[1: c_bits_num])

    # Draw circuit
    if verbose:
        circuit_plot = circuit.draw(output='text')
        print('\n{}'.format(circuit_plot))
    if store_results:
        circuit_filename = 'circuit_knn_{}.png'.format(init_modality)
        # circuit.draw(output='mpl', filename=os.path.join(res_dir, circuit_filename), fold=-1)
        # plt.close()

    # Execute the job
    if statevector:
        exec_modality = 'statevector'
        job = execute(circuit, backend)
    elif local_simulation:
        exec_modality = 'local_simulation'
        job = execute(circuit, backend, shots=shots)
    else:
        exec_modality = 'online_simulation' if online_simulation else 'quantum'
        job = execute(circuit, backend, shots=shots, job_name=job_name)

        # Check if the user wants to wait until the end of the execution
        if verbose:
            print(f'\nJob ID: {job.job_id()}')
        save_job_info(res_dir, job.job_id(), exec_modality, backend_name, init_gate, store_results, model='knn',
                      extra_args_list=[index_qubits, os.path.basename(training_data_file),
                      os.path.basename(normalized_data_files[0]), k, os.path.basename(normalized_data_files[1])],
                      verbose=verbose)

        if automatic_online_wait:
            wait_exec = True
        else:
            wait_exec = input('\nDo you want to wait until the end of the execution? [y/n] ')
            wait_exec = (wait_exec in ['y', 'Y'])

        if wait_exec:
            def handler(signal_received, frame):
                # Handle forced exit
                exit(0)

            signal(SIGINT, handler)

            job_monitor(job)
            # wait_for_job_completion(job, verbose=verbose)
        else:
            exit(0)

    # Get the results
    result = job.result()

    # Process the results depending on the execution modality
    if statevector:
        output_statevector = result.get_statevector(circuit, decimals=10)
        if verbose:
            print('\nOutput statevector:')
            print(list(output_statevector))

        p0, p1, index_cond_p, index_cond_p_diff_sorted = \
            process_output_statevector(output_statevector, qubits_num, index_qubits, N)

        if store_results:
            save_output_statevector(output_statevector, output_dir, init_modality, model='knn')
    else:
        counts = result.get_counts(circuit)
        sorted_counts = {i: counts[i] for i in sorted(counts)}
        if verbose:
            print('\nResults\nTotal output counts are: {}'.format(sorted_counts))

        if store_results:
            # Save counts data
            counts_filename = 'knn_counts_{}_{}'.format(exec_modality, init_modality)
            # Txt file
            with open(os.path.join(output_dir, counts_filename + '.txt'), 'w') as counts_file:
                counts_file.write(f'{counts}')
            # Histogram (unreadable for large circuits)
            # plot_histogram(counts)
            # plt.savefig(os.path.join(output_dir, counts_filename + '.pdf'))
            # plt.close()

        # Process counts
        p0, p1, index_cond_p, index_cond_p_diff_sorted =\
            process_counts(counts, index_qubits, N)

    # Get the k nearest neighbours
    nearest_neighbours_df = original_t_df.iloc[list(index_cond_p_diff_sorted.keys())[0: k], :]
    normalized_nearest_neighbours_df = t_df.iloc[list(index_cond_p_diff_sorted.keys())[0: k], :]

    # Display and (if required) store the results
    if verbose:
        print_knn_qcircuit_results(p0, p1, index_cond_p, index_cond_p_diff_sorted, k)

    normalized_knn_filename = None
    if store_results:
        log_filename = os.path.join(output_dir, 'knn_{}_{}_log.txt'.format(exec_modality, init_modality))
        save_knn_qcircuit_log(log_filename, p0, p1, index_cond_p, index_cond_p_diff_sorted, k)

        knn_filename = os.path.join(output_dir, 'knn_{}_{}.csv'.format(exec_modality, init_modality))
        nearest_neighbours_df.to_csv(knn_filename, index=False)

        normalized_knn_filename = os.path.join(output_dir, 'normalized_knn_{}_{}.csv'.format(exec_modality,
                                                                                             init_modality))
        normalized_nearest_neighbours_df.to_csv(normalized_knn_filename, index=False)

    # Display the final outcome
    if verbose:
        print(f'\nThe normalized {k} nearest neighbours for the unclassified instance provided are:')
        print(normalized_nearest_neighbours_df)

    normalized_nearest_neighbours_df.reset_index(drop=True, inplace=True)
    return normalized_nearest_neighbours_df, ui_df, normalized_knn_filename, normalized_data_files[1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for executing the KNN algorithm, either classical or quantum'
                                                 ' (the default execution modality is quantum).')
    parser.add_argument('training_data_file', metavar='training_data_file', type=str, nargs='?',
                        help='file containing training data for the KNN (csv file)')
    parser.add_argument('unclassified_instance_file', metavar='unclassified_instance_file', type=str, nargs='?',
                        help='file containing the unclassified instance (csv file)')
    parser.add_argument('--attribute-normalization', dest='attribute_normalization', action='store_const',
                        const=True, default=False, help='normalize the attributes by summing the minimum and dividing '
                                                        'by the range')
    parser.add_argument('k', metavar='k', type=int, nargs='?', help='number of nearest neighbours'
                                                                    ' (KNN hyper-parameter)')
    parser.add_argument('--classical', dest='classical', action='store_const',
                        const=True, default=False, help='execute the classical version of KNN')
    parser.add_argument('--statevector', dest='statevector', action='store_const',
                        const=True, default=False, help='execute the quantum KNN locally using Aer\'s simulator and '
                        'process the final statevector instead of sampling from it (this parameter is considered '
                        'only if the classical flag is disabled)')
    parser.add_argument('--local-simulation', dest='local_simulation', action='store_const',
                        const=True, default=False, help='execute the quantum KNN locally using Aer\'s simulator '
                        '(this parameter is considered only if the classical and the statevector flags are disabled)')
    parser.add_argument('--online-simulation', dest='online_simulation', action='store_const',
                        const=True, default=False, help='execute the quantum KNN online using an IBMQ\'s '
                        'simulator (this parameter is considered only if the classical, the statevector and the '
                        'local-simulation flags are disabled)')
    parser.add_argument('--backend-name', metavar='backend_name', type=str, nargs='?',
                        default='ibmq_qasm_simulator|ibmq_lima', help='name of the online backend, either a quantum '
                        'device or an online simulator (this parameter is considered only if the classical, the '
                        'statevector and the simulation-local flags are disabled)')
    parser.add_argument('--job-name', metavar='job_name', type=str, nargs='?',
                        default=None, help='name assigned to the job (used only for the execution on an online backend')
    parser.add_argument('--shots', metavar='shots', type=int, nargs='?',
                        default=1024, help='number of shots (for simulations and quantum executions)')
    parser.add_argument('--res-dir', metavar='res_dir', type=str, nargs='?',
                        default=None, help='directory where to store the results')
    parser.add_argument('--not-store', dest='not_store', action='store_const',
                        const=True, default=False, help='The results of the execution are not stored in memory.')
    parser.add_argument('--not-verbose', dest='not_verbose', action='store_const',
                        const=True, default=False, help='No information is printed on the stdout (except for the '
                                                        'request to wait or not the end of the online execution).')
    args = parser.parse_args()

    root_res_dir = args.res_dir if args.res_dir is not None \
        else os.path.join('./', os.path.dirname(sys.argv[0]), 'results')
    if args.classical:
        prefix = 'classical'
    elif args.statevector:
        prefix = 'statevector'
    elif args.local_simulation:
        prefix = 'local_simulation'
    elif args.online_simulation:
        prefix = 'online_simulation'
    else:
        prefix = 'quantum'
    res_dir = os.path.join(root_res_dir, '{}_{}'.format(prefix, datetime.now().strftime('%d-%m-%Y_%H-%M-%S')))

    init_gate = False

    backend_name = args.backend_name
    if not args.classical and not args.statevector and not args.local_simulation and\
            backend_name == 'ibmq_qasm_simulator|ibmq_lima':
        backend_name = backend_name.split('|')[0 if args.online_simulation else 1]

    run_knn(args.training_data_file, args.unclassified_instance_file, args.attribute_normalization, args.k,
            args.classical, args.statevector, args.local_simulation, args.online_simulation, backend_name,
            args.job_name, init_gate, args.shots, res_dir, not args.not_store, not args.not_verbose)
