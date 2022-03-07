import argparse
import json
import numpy as np
import os
import pandas as pd


def preprocess_dataset(dataset_file, sep, no_header, new_header, output_instances_num, output_classes_num, class_column,
                       classes_of_interest, convert_classes_into, delete_columns, out_file):
    dataset = pd.read_csv(dataset_file, sep=sep, header=(None if no_header else 'infer'))

    if no_header and new_header is None:
        print('A new header must be provided since the input dataset does not contain one!')
        exit(0)
    if new_header is not None and \
            len(new_header) != (len(dataset.columns) - (len(delete_columns) if delete_columns is not None else 0)):
        print('The number of elements of the new header is not correct!')
        exit(0)

    if class_column is None:
        class_column = len(dataset.columns) - 1
    dataset.iloc[:, class_column] = dataset.iloc[:, class_column].astype(str)
    initial_classes = pd.unique(dataset.iloc[:, class_column])

    if len(initial_classes) > output_classes_num:
        if classes_of_interest is not None:
            if len(classes_of_interest) == output_classes_num:
                dataset = dataset.loc[dataset.iloc[:, class_column].isin(classes_of_interest)]
            else:
                print('Too many classes of interest w.r.t. the desired output number!')
                exit(0)
        else:
            class_values_series = dataset.iloc[:, class_column].value_counts()
            out_classes = [x for x, _ in class_values_series.iteritems()][0:output_classes_num]
            dataset = dataset.loc[dataset.iloc[:, class_column].isin(out_classes)]
        dataset = dataset.reset_index(drop=True)
    elif len(initial_classes) < output_classes_num:
        print('Not enough classes w.r.t. the desired number!')
        exit(0)

    replace_dict = None
    if convert_classes_into is not None:
        if len(convert_classes_into) == output_classes_num:
            if classes_of_interest is not None:
                class_labels = classes_of_interest
            else:
                class_labels = sorted(pd.unique(dataset.iloc[:, class_column]))
            replace_dict = {old_l: new_l for old_l, new_l in zip(class_labels, convert_classes_into)}
            dataset.iloc[:, class_column].replace(replace_dict, inplace=True)
        else:
            print('The number of labels provided in \'convert-classes-into\' does not match the number of output'
                  'classes (output-classes-num)!')
            exit(0)

    instances_num = len(dataset)
    output_instances_num = output_instances_num if output_instances_num is not None else instances_num
    if instances_num > output_instances_num:
        dataset = dataset.groupby(dataset.columns[class_column]).\
            apply(pd.DataFrame.sample, frac=(output_instances_num/instances_num))\
            .reset_index(drop=True)  # random_state=x for reproducibility of sampling

    if delete_columns is not None and class_column in delete_columns:
        print('Operation not allowed: you are trying to delete the class column!')
        exit(0)

    if class_column != (len(dataset.columns) - 1):
        columns = list(dataset.columns)
        columns += [columns.pop(class_column)]
        dataset = dataset[columns]

        if delete_columns is not None:
            delete_columns = [x if x < class_column else x-1 for x in delete_columns]

    if delete_columns is not None:
        dataset = dataset.drop(dataset.columns[delete_columns], axis=1)

    if out_file is None:
        out_file = '{}_preprocessed.csv'.format(os.path.splitext(dataset_file)[0])
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    if replace_dict is not None:
        class_mapping_file = '{}_class_mapping.json'.format(os.path.splitext(out_file)[0])
        if type(list(replace_dict.keys())[0]).__module__ == np.__name__:
            replace_dict = {key.item(): val for key, val in replace_dict.items()}
        with open(class_mapping_file, 'w') as cmf:
            json.dump(replace_dict, cmf, ensure_ascii=False, indent=4)

    dataset.to_csv(out_file, sep=',', header=(True if new_header is None else new_header), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for prepocessing datasets in order to make them suitable for '
                                                 'the models pipeline.')
    parser.add_argument('dataset', metavar='dataset', type=str, nargs='?', default=None,
                        help='file representing the dataset')
    parser.add_argument('--sep', metavar='sep', type=str, nargs='?', default=',',
                        help='separator used in the dataset file (\',\' is used by default)')
    parser.add_argument('--no-header', dest='no_header', action='store_const',
                        const=True, default=False, help='the input dataset file does not contain the header')
    parser.add_argument('--new-header', metavar='new_header', type=str, nargs='+', default=None,
                        help='header put in the output file (it must be used when there is no header)')
    parser.add_argument('--output-instances-num', metavar='output_instances_num', type=int, nargs='?', default=None,
                        help='number of output instances')
    parser.add_argument('--output-classes-num', metavar='output_classes_num', type=int, nargs='?', default=2,
                        help='number of output classes (2 is used by default)')
    parser.add_argument('--class-column', metavar='class_column', type=int, nargs='?', default=None,
                        help='index (between 0 and len(columns)) of the class column (the last one is used by default)')
    parser.add_argument('--classes-of-interest', metavar='classes_of_interest', type=str, nargs='+', default=None,
                        help='list of classes of interest (the most represented ones are kept by default)')
    parser.add_argument('--convert-classes-into', metavar='convert_classes_into', type=str, nargs='+', default=None,
                        help='list of classes to be put in the output file')
    parser.add_argument('--delete-columns', metavar='delete_columns', type=int, nargs='+', default=None,
                        help='list of indices of columns to delete')
    parser.add_argument('--out-file', metavar='out_file', type=str, nargs='?', default=None,
                        help='output filename')
    args = parser.parse_args()
    
    preprocess_dataset(args.dataset, args.sep, args.no_header, args.new_header, args.output_instances_num,
                       args.output_classes_num, args.class_column, args.classes_of_interest,
                       args.convert_classes_into, args.delete_columns, args.out_file)
