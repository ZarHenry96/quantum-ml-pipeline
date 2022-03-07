from qiskit import (
    ClassicalRegister,
    QuantumRegister,
    QuantumCircuit,
    execute,
    Aer,
    IBMQ
)
from qiskit.circuit.library import XGate, MCMT
from qiskit.providers.jobstatus import JOB_FINAL_STATES
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram

import argparse
import math
import numpy as np
import os
import pandas as pd
import shutil
import time
import sys

from datetime import datetime, timezone
from matplotlib import pyplot as plt
from signal import signal, SIGINT

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_data(training_data_file, unclassified_instance_file, attribute_normalization, instance_normalization,
              copy_input_files, res_dir, show=True):
    t_df = pd.read_csv(training_data_file, sep=',')
    ui_df = pd.read_csv(unclassified_instance_file, sep=',')

    if copy_input_files:
        shutil.copy2(training_data_file, res_dir)
        shutil.copy2(unclassified_instance_file, res_dir)

    if (len(t_df.columns)-1) != len(ui_df.columns):
        print('Error: dimensionality of training data and unclassified instance do not match!', file=sys.stderr)
        exit(-1)

    features_number = len(t_df.columns) - 1
    if attribute_normalization:
        # normalize attributes by subtracting the minimum and dividing by the range
        training_min = t_df.iloc[:, :features_number].min()
        training_range = t_df.iloc[:, :features_number].max() - t_df.iloc[:, :features_number].min()

        # replace the zero range values with 1
        zero_range_columns = np.nonzero(~(training_range.to_numpy() != 0))[0]
        if len(zero_range_columns) > 0:
            training_range.iloc[zero_range_columns] = [1 for _ in range(0, len(zero_range_columns))]

        ui_df.iloc[:, :features_number] = (ui_df.iloc[:, :features_number] - training_min) / training_range
        for attribute_index, attribute_val in enumerate(ui_df.iloc[0, 0:features_number]):
            ui_df.iloc[0, attribute_index] = max(min(1, attribute_val), 0)

        t_df.iloc[:, :features_number] = (t_df.iloc[:, :features_number] - training_min) / training_range

    if instance_normalization:
        # replace zero rows with epsilon values
        eps_val = 0.000001
        t_df_zero_rows = t_df[(t_df.iloc[:, 0:features_number] == 0).all(axis=1)].index.values.tolist()
        if len(t_df_zero_rows) > 0:
            t_df.iloc[t_df_zero_rows, 0:features_number] = [eps_val for _ in range(0, features_number)]
        ui_df_zero_rows = ui_df[(ui_df.iloc[:, 0:features_number] == 0).all(axis=1)].index.values.tolist()
        if len(ui_df_zero_rows):
            ui_df.iloc[ui_df_zero_rows, 0:features_number] = [eps_val for _ in range(0, features_number)]

        # normalize data vectors dividing each of them by its norm
        t_df.iloc[:, :features_number] = t_df.iloc[:, :features_number]\
            .div(np.sqrt(np.square(t_df.iloc[:, :features_number]).sum(axis=1)), axis=0)
        ui_df.iloc[:, :features_number] = ui_df.iloc[:, :features_number]\
            .div(np.sqrt(np.square(ui_df.iloc[:, :features_number]).sum(axis=1)), axis=0)

    if show:
        print('Normalized training dataset:')
        print(t_df)
        print('\nNormalized unclassified instance:')
        print(ui_df)
        print()

    normalized_data_files = [None, None]
    if (attribute_normalization or instance_normalization) and copy_input_files:
        normalized_training_data_file = \
            os.path.join(res_dir, 'normalized_{}'.format(os.path.basename(training_data_file)))
        t_df.to_csv(
            normalized_training_data_file,
            index=False
        )

        normalized_unclassified_instance_file = \
            os.path.join(res_dir, 'normalized_{}'.format(os.path.basename(unclassified_instance_file)))
        ui_df.to_csv(
            normalized_unclassified_instance_file,
            index=False
        )

        normalized_data_files = [normalized_training_data_file, normalized_unclassified_instance_file]

    return t_df, ui_df, normalized_data_files


def select_backend(statevector, local_simulation, online_simulation, backend_name):
    if statevector or local_simulation:
        # Use the Aer simulator
        backend = Aer.get_backend('aer_simulator')
    else:
        # Use IBMQ backends
        if IBMQ.active_account() is not None:
            IBMQ.disable_account()
        provider = IBMQ.load_account()
        backend = provider.get_backend(backend_name)

    return backend


def classical_classifier(t_df, ui_df, save_to_file, res_dir, expectation=False, verbose=True):
    output_dir = os.path.join(res_dir, 'classical_expectation' if expectation else 'output')
    if save_to_file:
        os.makedirs(output_dir, exist_ok=True)

    sum_val = 0
    for instance_index, row in t_df.iterrows():
        sum_term = 0
        for t_f_ampl, ui_f_ampl in zip(row[0:-1], ui_df.iloc[0, :]):
            sum_term += t_f_ampl*ui_f_ampl
        sum_term *= row[-1]
        sum_val += sum_term

    label = int(np.sign(sum_val))

    verb_to_print = 'expected' if expectation else 'predicted'
    if verbose:
        print(f'Classically {verb_to_print} label: {label}\t(value={sum_val})')

    if save_to_file:
        prefix = 'expected_' if expectation else ''
        postfix = '(classical)' if expectation else 'classical'
        with open(os.path.join(output_dir, f'{prefix}output_label_{postfix}.txt'), 'w') \
                as classical_cl_out:
            classical_cl_out.write(f'Label of unclassified instance: {label}\t(value={sum_val})')

    return label


def joint_initialization(circuit, qr, qubits_num, index_qubits, features_qubits,
                         t_df, ui_df, N, d):
    # Hadamard gates for swap test
    circuit.h(qr[0])
    circuit.h(qr[1])

    # Ancillary control qubit
    ancillary_control_qubit = 2

    # Initialize jointly third swap test qubit, index register and features register
    init_qubits = 1 + index_qubits + features_qubits
    amplitudes = np.zeros(2 ** init_qubits)
    amplitude_base_value = 1.0 / math.sqrt(2 * N)

    # Training data amplitudes
    for instance_index, row in t_df.iterrows():
        for feature_indx, feature_amplitude in enumerate(row[0:-1]):
            index = 2 * instance_index + (2 ** (index_qubits+1)) * feature_indx
            amplitudes[index] = amplitude_base_value * feature_amplitude

    # Unclassified instance amplitudes
    for i in range(0, N):
        for feature_indx, feature_amplitude in enumerate(ui_df.iloc[0, 0:d]):
            index = 1 + 2 * i + (2 ** (index_qubits+1)) * feature_indx
            amplitudes[index] = amplitude_base_value * feature_amplitude

    # Set all ancillary_qubit+index_register+features_register amplitudes
    circuit.initialize(amplitudes, qr[ancillary_control_qubit: ancillary_control_qubit + init_qubits])

    # Set training data labels
    circuit.x(qr[ancillary_control_qubit])
    for instance_index, row in t_df.iterrows():
        label = row[d]
        if label == -1:
            bin_indx = ('{0:0' + str(index_qubits) + 'b}').format(instance_index)
            zero_qubits_indices = [
                ancillary_control_qubit + len(bin_indx) - i
                for i, letter in enumerate(bin_indx) if letter == '0'
            ]

            # select the right qubits from the index register
            for qubit_indx in zero_qubits_indices:
                circuit.x(qr[qubit_indx])

            # add multi controlled CNOT gate
            multi_controlled_cnot = MCMT(XGate(), index_qubits + 1, 1)
            circuit.compose(multi_controlled_cnot,
                            qr[ancillary_control_qubit: ancillary_control_qubit + index_qubits + 1] + [
                                qr[qubits_num - 1]],
                            inplace=True)

            # bring the index register qubits back to the original state
            for qubit_indx in zero_qubits_indices:
                circuit.x(qr[qubit_indx])
    circuit.x(qr[ancillary_control_qubit])

    # Set unclassified instance labels
    circuit.cx(qr[ancillary_control_qubit], qr[qubits_num - 1])
    circuit.ch(qr[ancillary_control_qubit], qr[qubits_num - 1])

    # Add swap test gates
    circuit.cswap(qr[0], qr[1], qr[2])
    circuit.h(qr[0])


def process_statevector(output_statevector):
    # Compute and display the probability of measuring 0 and 1 on the first qubit
    p0, p1 = 0, 0
    for i, amplitude in enumerate(output_statevector):
        if i % 2 == 0:
            p0 += (np.abs(amplitude) ** 2)
        else:
            p1 += (np.abs(amplitude) ** 2)

    return p0, p1


def save_output_statevector(output_statevector, res_dir, init_modality, model='classifier'):
    with open(os.path.join(res_dir, f'{model}_output_statevector_statevector_{init_modality}.txt'), 'w') \
            as output_statevector_file:
        output_statevector_file.write(f'{list(output_statevector)}')


def save_statevector_probabilities(p0, p1, res_dir, init_modality):
    with open(os.path.join(res_dir, f'classifier_probabilities_statevector_{init_modality}.txt'), 'w') \
            as statevector_prob_file:
        statevector_prob_file.write(f'Probability of measuring 0:\t{p0}\n')
        statevector_prob_file.write(f'Probability of measuring 1:\t{p1}\n')


def save_job_info(res_dir, job_id, exec_modality, backend_name, init_gate, store_results,
                  model='classifier', extra_args_list=None, verbose=True):
    if store_results:
        filename = os.path.join(res_dir, f'{model}_job_info.txt')
        if verbose:
            print('The Job ID will be saved in {}'.format(filename))
        with open(filename, 'w') as job_info_file:
            job_info_file.write(f'{job_id}\n')
            job_info_file.write(f'{exec_modality}\n')
            job_info_file.write(f'{backend_name}\n')
            initialization_mode = 'init_gate' if init_gate else 'joint_init'
            job_info_file.write(f'{initialization_mode}\n')

            if extra_args_list is not None:
                for entry in extra_args_list:
                    job_info_file.write(f'{entry}\n')


def wait_for_job_completion(job, verbose=True):
    start_time = datetime.now(timezone.utc)
    job_status = job.status()
    while job_status not in JOB_FINAL_STATES:
        current_time = datetime.now(timezone.utc)
        exec_time = str(current_time - start_time).split('.')[0]

        status = job_status.name

        queue_info = job.queue_info()
        if queue_info is None:
            position, run_time = None, None
        else:
            position = queue_info.position
            run_time = str(queue_info.estimated_complete_time - current_time).split('.')[0] \
                if current_time < queue_info.estimated_complete_time \
                else 'new estimation in progress'

        if verbose:
            print(f'Status @ {exec_time} s: {status},'
                  f' est. queue position: {position},'
                  f' est. run time: {run_time}')

        time.sleep(60)
        job_status = job.status()


def run_classifier(training_data_file, unclassified_instance_file, attribute_normalization, classical, statevector,
                   local_simulation, online_simulation, backend_name, job_name, init_gate, shots,
                   res_dir, store_results, verbose, already_normalized=False, automatic_online_wait=False):
    input_dir = os.path.join(res_dir, 'input')
    output_dir = os.path.join(res_dir, 'output')
    if store_results:
        os.makedirs(res_dir, exist_ok=True)
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

    # get data from input files
    if already_normalized:
        attribute_normalization, instance_normalization = False, False
    else:
        instance_normalization = True
    t_df, ui_df, _ = load_data(training_data_file, unclassified_instance_file, attribute_normalization,
                               instance_normalization, store_results, input_dir, show=verbose)
    N, d = len(t_df), len(t_df.columns)-1

    if classical:
        # Predict the label value
        return classical_classifier(t_df, ui_df, store_results, res_dir, verbose=verbose)

    # Select the backend for the execution
    backend = select_backend(statevector, local_simulation, online_simulation, backend_name)

    # Compute the expected label value (classical computation)
    classical_classifier(t_df, ui_df, store_results, res_dir, expectation=True, verbose=verbose)

    # compute circuit size
    swap_circuit_qubits = 3
    index_qubits = math.ceil(math.log2(N))
    features_qubits = math.ceil(math.log2(d))
    label_qubits = 1
    qubits_num = swap_circuit_qubits+index_qubits+features_qubits+label_qubits

    # Create a Quantum Circuit acting on the q register
    qr = QuantumRegister(qubits_num, 'q')
    cr = ClassicalRegister(1, 'c')
    circuit = QuantumCircuit(qr, cr)

    # Build the circuit using the selected method
    if not init_gate:
        init_modality = 'joint_init'
        joint_initialization(circuit, qr, qubits_num, index_qubits, features_qubits, t_df, ui_df, N, d)
    else:
        init_modality = 'init_gate'
        print('Not implemented yet', file=sys.stderr)
        exit(0)

    # Measure qubit for swap test
    if statevector:
        circuit.save_statevector()
    else:
        circuit.measure([0], [0])

    # Draw circuit
    if verbose:
        circuit_plot = circuit.draw(output='text')
        print('\n{}'.format(circuit_plot))
    if store_results:
        circuit_filename = 'classifier_circuit_{}.png'.format('init_gate' if init_gate else 'joint_init')
        # circuit.draw(output='mpl', filename=os.path.join(res_dir, circuit_filename), fold=-1)
        # plt.close()

    # Execute the simulation
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
        save_job_info(res_dir, job.job_id(), exec_modality, backend_name, init_gate, store_results, verbose=verbose)

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

    # Get results
    result = job.result()

    # Process the results depending on the execution modality
    if statevector:
        output_statevector = result.get_statevector(circuit, decimals=10)
        if verbose:
            print('\nOutput statevector:')
            print(list(output_statevector))

        p0, p1 = process_statevector(output_statevector)

        # Save the statevector and the probabilities to file (if required)
        if store_results:
            save_output_statevector(output_statevector, output_dir, init_modality)
            save_statevector_probabilities(p0, p1, output_dir, init_modality)
    else:
        counts = result.get_counts(circuit)
        sorted_counts = {i: counts[i] for i in sorted(counts)}
        if verbose:
            print('\nResults\nTotal output counts are: {}'.format(sorted_counts))

        # Save counts to txt file and as histogram
        if store_results:
            counts_filename = 'classifier_counts_{}_{}'.format(exec_modality, init_modality)
            # Txt file
            with open(os.path.join(output_dir, counts_filename+'.txt'), 'w') as counts_file:
                counts_file.write(f'{counts}')
            # Histogram (unreadable for large circuits)
            # plot_histogram(counts)
            # plt.savefig(os.path.join(output_dir, counts_filename+'.pdf'))
            # plt.close()

        # Compute the probability of measuring 0 and 1
        p0 = counts['0']/shots
        p1 = counts['1']/shots

    # Display the probabilities
    if verbose:
        print(f'\nProbability of measuring 0: {p0}')
        print(f'Probability of measuring 1: {p1}')

    # Predict and display the label
    ui_label = -1 if p1 > 0.25 else 1
    if verbose:
        print(f'\nLabel of unclassified instance: {ui_label}\t(value={1-4*p1})')

    if store_results:
        label_filename = 'output_label_{}_{}.txt'.format(exec_modality, init_modality)
        with open(os.path.join(output_dir, label_filename), 'w') as label_file:
            label_file.write(f'Label of unclassified instance: {ui_label}\t(value={1-4*p1})')

    return ui_label


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for executing a binary classifier based on cosine similarity, '
                                     'either classical or quantum (Pastorello-Blanzieri\'s Binary Classifier). The '
                                     'default execution modality is quantum.')
    parser.add_argument('training_data_file', metavar='training_data_file', type=str, nargs='?',
                        help='file containing training data for the classifier (csv file)')
    parser.add_argument('unclassified_instance_file', metavar='unclassified_instance_file', type=str, nargs='?',
                        help='file containing the unclassified instance (csv file)')
    parser.add_argument('--attribute-normalization', dest='attribute_normalization', action='store_const',
                        const=True, default=False, help='normalize the attributes by summing the minimum and dividing '
                                                        'by the range')
    parser.add_argument('--classical', dest='classical', action='store_const',
                        const=True, default=False, help='execute the classical version of the classifier')
    parser.add_argument('--statevector', dest='statevector', action='store_const',
                        const=True, default=False, help='execute the quantum classifier locally using Aer\'s simulator '
                        ' and process the final statevector instead of sampling from it (this parameter is considered '
                        'only if the classical flag is disabled)')
    parser.add_argument('--local-simulation', dest='local_simulation', action='store_const',
                        const=True, default=False, help='execute the quantum classifier locally using Aer\'s simulator '
                        '(this parameter is considered only if the classical and the statevector flags are disabled)')
    parser.add_argument('--online-simulation', dest='online_simulation', action='store_const',
                        const=True, default=False, help='execute the quantum classifier online using an IBMQ\'s '
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

    run_classifier(args.training_data_file, args.unclassified_instance_file, args.attribute_normalization,
                   args.classical, args.statevector, args.local_simulation, args.online_simulation, backend_name,
                   args.job_name, init_gate, args.shots, res_dir, not args.not_store, not args.not_verbose)
