#!/usr/bin/python3

"""create_baseline_datasets.py

The baselne models are not capable of modelling sequential data. For these
models, hand-crafted features are obtained from the timeseries.
"""

__author__ = "Bas Straathof"

import argparse, json, os

from sys import argv

import numpy as np

from sklearn.preprocessing import StandardScaler

from ..utils.preprocessing_utils import baseline_features_targets, \
        compute_statistics, get_subseq
from ..utils.utils import get_subject_dirs


def parse_cl_args():
    """Parses CL arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-sp', '--subjects-path', type=str, default='data/',
            help='Path to subject directories.')
    parser.add_argument('-v', '--verbose', type=str, default=True,
            help='Console output verbosity flag.')

    return parser.parse_args(argv[1:])


def main(args):
    if not os.path.exists('data/baseline_features/'):
        os.makedirs('data/baseline_features/')
        train_dirs = get_subject_dirs(os.path.join(args.subjects_path,
            'train/'))
        test_dirs = get_subject_dirs(os.path.join(args.subjects_path, 'test/'))
        scaler = StandardScaler()

        with open('nicu_los/config.json') as f:
            config = json.load(f)
            variables = config['variables']
            sub_seqs = config['baseline_subsequences']

        # Get training features and targets
        X_train, y_train, subject_ids_train = baseline_features_targets(
                train_dirs, variables, sub_seqs)

        # Get test features and targets
        X_test, y_test, subject_ids_test = baseline_features_targets(
                test_dirs, variables, sub_seqs)

        # Normalize X_train
        X_train = scaler.fit_transform(X_train)

        # Normalize X_test
        X_test = scaler.transform(X_test)

        np.save('data/baseline_features/X_train', X_train)
        np.save('data/baseline_features/Y_train', y_train)
        np.save('data/baseline_features/subject_ids_train', subject_ids_train)
        np.save('data/baseline_features/X_test', X_test)
        np.save('data/baseline_features/Y_test', y_test)
        np.save('data/baseline_features/subject_ids_test', subject_ids_test)

if __name__ == '__main__':
    main(parse_cl_args())

