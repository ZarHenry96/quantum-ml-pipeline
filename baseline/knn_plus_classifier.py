import numpy as np
import os

import pandas as pd
from scipy.spatial.distance import cosine
from sklearn.neighbors import NearestNeighbors


class KNNPlusClassifier(object):
    def __init__(self, k, knn_metric=cosine, res_dir=None):
        super().__init__()
        self.k = k
        self.knn_metric = knn_metric
        self.knn_model = NearestNeighbors(n_neighbors=self.k, metric=self.knn_metric, algorithm='brute')

        self.res_dir = res_dir
        if self.res_dir is not None:
            os.makedirs(res_dir, exist_ok=True)

        self.training_data = None
        self.training_labels = None

    def fit(self, training_data, training_labels):
        self.training_data = np.array(training_data)
        if isinstance(training_data, pd.DataFrame):
            self.columns = list(training_data.columns)
        else:
            self.columns = None
        self.training_labels = np.array(training_labels)
        self.knn_model.fit(self.training_data)

    def predict(self, test_data):
        np_test_data = np.array(test_data)
        test_data_kneighbors = self.knn_model.kneighbors(np_test_data)

        predicted_labels = []
        for j, (distances, indices, test_instance) in \
                enumerate(zip(test_data_kneighbors[0], test_data_kneighbors[1], np_test_data)):
            nearest_neighbors = self.training_data[indices, :]
            nearest_neighbors_labels = self.training_labels[indices]

            if self.res_dir is not None:
                self.save_knn_out(distances, indices, nearest_neighbors, nearest_neighbors_labels,
                                  os.path.join(self.res_dir, 'test_{}'.format(j)))

            sum_val = 0
            for training_instance, training_label in zip(nearest_neighbors, nearest_neighbors_labels):
                sum_term = 0
                for training_instance_ampl, test_instance_ampl in zip(training_instance, test_instance):
                    sum_term += training_instance_ampl * test_instance_ampl
                sum_term /= (np.linalg.norm(training_instance) * np.linalg.norm(test_instance))
                sum_term *= training_label
                sum_val += sum_term

            label = int(np.sign(sum_val))
            predicted_labels.append(label)

            if self.res_dir is not None:
                self.save_classifier_out(label, sum_val, os.path.join(self.res_dir, 'test_{}'.format(j)))

        return predicted_labels

    def save_knn_out(self, distances, indices, nearest_neighbors, nearest_neighbors_labels, out_dir):
        os.makedirs(out_dir, exist_ok=True)

        with open(os.path.join(out_dir, 'knn_log.txt'), 'w') as knn_log:
            knn_log.write(f'Classically predicted {self.k} nearest neighbours:')
            for distance, index, nn in zip(distances, indices, nearest_neighbors):
                element = np.array2string(nn, separator=', ')
                knn_log.write(f'\n\tdistance: {distance}, index: {index}, element: {element}')

        knn_filename = os.path.join(out_dir, 'normalized_knn.csv')
        complete_df = pd.concat(
            [pd.DataFrame(nearest_neighbors, columns=self.columns),
             pd.Series(nearest_neighbors_labels, name='class')], axis=1)
        complete_df.to_csv(knn_filename, index=False)

    def save_classifier_out(self, label, sum_val, out_dir):
        with open(os.path.join(out_dir, 'output_label.txt'), 'w') \
                as classical_cl_out:
            classical_cl_out.write(f'Label of unclassified instance: {label}\t(value={sum_val})')
