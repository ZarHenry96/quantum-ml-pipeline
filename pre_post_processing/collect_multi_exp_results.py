import argparse
import json
import os
import pandas as pd


class MyJSONEncoder(json.JSONEncoder):
    def __init__(self, *args, **kwargs):
        super(MyJSONEncoder, self).__init__(*args, **kwargs)
        self.current_indent = 0
        self.current_indent_str = ""

    def encode(self, obj):
        if isinstance(obj, (list, tuple)):
            primitives_only = True
            for item in obj:
                if isinstance(item, (list, tuple, dict)):
                    primitives_only = False
                    break
            output = []
            if primitives_only:
                for item in obj:
                    output.append(json.dumps(item))
                return "[" + ", ".join(output) + "]"
            else:
                self.current_indent += self.indent
                self.current_indent_str = "".join([" " for x in range(self.current_indent)])
                for item in obj:
                    output.append(self.current_indent_str + self.encode(item))
                self.current_indent -= self.indent
                self.current_indent_str = "".join([" " for x in range(self.current_indent)])
                return "[\n" + ",\n".join(output) + "\n" + self.current_indent_str + "]"
        elif isinstance(obj, dict):
            output = []
            self.current_indent += self.indent
            self.current_indent_str = "".join([" " for x in range(self.current_indent)])
            for key, value in obj.items():
                output.append(self.current_indent_str + json.dumps(key) + ": " + self.encode(value))
            self.current_indent -= self.indent
            self.current_indent_str = "".join([" " for x in range(self.current_indent)])
            return "{\n" + ",\n".join(output) + "\n" + self.current_indent_str + "}"
        else:
            return json.dumps(obj)


def process_res_dirs(res_dirs, baseline, out_dir, out_files):
    results_per_dataset = {}
    results_per_method = {}

    # If the directories are related to baseline methods, process subdirs of res_dirs
    if not baseline:
        dirs_to_process = res_dirs
    else:
        dirs_to_process = []
        for directory in res_dirs:
            dirs_to_process += [
                os.path.join(directory, d)
                for d in os.listdir(directory)
                if os.path.isdir(os.path.join(directory, d))
            ]

    # Process results directories
    for res_dir in dirs_to_process:
        # Load experiment config file and extract relevant information
        if not baseline:
            with open(os.path.join(res_dir, 'exp_config.json')) as ecf:
                exp_config = json.load(ecf)
            dataset_name = os.path.basename(exp_config['dataset']).rsplit('.', 1)[0]
            method = '{}_{}'.format(exp_config['knn']['exec_type'], exp_config['classifier']['exec_type'])
        else:
            with open(os.path.join(os.path.dirname(res_dir), 'baseline_config.json')) as bcf:
                baseline_config = json.load(bcf)
            dataset_name = os.path.basename(baseline_config['dataset']).rsplit('.', 1)[0]
            method = os.path.basename(res_dir)

        # Load results .csv file
        exp_results = pd.read_csv(os.path.join(res_dir, 'results.csv'), sep=',')

        # Compute accuracy per fold
        accuracy_per_fold = exp_results.groupby('fold') \
            .apply(lambda x: (x['expected_label'] == x['predicted_label']).sum() / len(x))
        accuracy_per_fold_values = list(accuracy_per_fold.values)

        # Insert data in results dictionaries
        if dataset_name in results_per_dataset:
            results_per_dataset[dataset_name][method] = accuracy_per_fold_values
        else:
            results_per_dataset[dataset_name] = {method: accuracy_per_fold_values}

        if method in results_per_method:
            results_per_method[method][dataset_name] = accuracy_per_fold_values
        else:
            results_per_method[method] = {dataset_name: accuracy_per_fold_values}

    # Sort dictionaries by key
    results_per_dataset = {
        k: {k1: results_per_dataset[k][k1] for k1 in sorted(results_per_dataset[k])}
        for k in sorted(results_per_dataset)
    }
    results_per_method = {
        k: {k1: results_per_method[k][k1] for k1 in sorted(results_per_method[k])}
        for k in sorted(results_per_method)
    }

    # Set the output directory
    if out_dir is not None:
        output_directory = out_dir
        os.makedirs(output_directory, exist_ok=True)
    else:
        output_directory = os.path.dirname(res_dirs[0])

    # Set the output filenames
    if out_files is None or len(out_files) != 2:
        output_filenames = ['results_per_dataset.json', 'results_per_method.json']
    else:
        output_filenames = out_files

    # Save results
    with open(os.path.join(output_directory, output_filenames[0]), 'w') as rpdf:
        rpdf.write(json.dumps(results_per_dataset, cls=MyJSONEncoder, ensure_ascii=False, indent=4))

    with open(os.path.join(output_directory, output_filenames[1]), 'w') as rpmf:
        rpmf.write(json.dumps(results_per_method, cls=MyJSONEncoder, ensure_ascii=False, indent=4))


def process_root_res_dir(root_res_dir, baseline, out_dir, out_files):
    res_dirs = [
        os.path.join(root_res_dir, d)
        for d in os.listdir(root_res_dir)
        if os.path.isdir(os.path.join(root_res_dir, d))
    ]
    process_res_dirs(res_dirs, baseline, out_dir, out_files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for collecting the results of multiple experiments')
    parser.add_argument('--res-dirs', metavar='res_dirs', type=str, nargs='+', default=None,
                        help='list of directories containing the results of the experiments')
    parser.add_argument('--root-res-dir', metavar='root_res_dir', type=str, nargs='?', default=None,
                        help='directory containing the results directories of the experiments')
    parser.add_argument('--baseline', dest='baseline', action='store_const', const=True, default=False,
                        help='the results directories provided are related to baseline methods')
    parser.add_argument('--out-dir', metavar='out_dir', type=str, action='store', nargs='?',
                        default=None, help='directory where to store the 2 output files (the parent directory of the'
                                           'first res-dir/the root-res-dir is used by default)')
    parser.add_argument('--out-files', metavar='out_files', type=str, action='store', nargs='+',
                        default=None, help='output filenames (they must be 2: per dataset/per method)')
    args = parser.parse_args()

    if args.res_dirs is not None:
        process_res_dirs(args.res_dirs, args.baseline, args.out_dir, args.out_files)
    elif args.root_res_dir is not None:
        process_root_res_dir(args.root_res_dir, args.baseline, args.out_dir, args.out_files)
