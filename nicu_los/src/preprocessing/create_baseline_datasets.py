#!/usr/bin/python3

"""create_baseline_datasets.py

The baselne models are not capable of modelling sequential data. For these
models, hand-crafted features are obtained from the timeseries.
"""

__author__ = "Bas Straathof"

import argparse, json, os

from sys import argv

import numpy as np
import multiprocessing as mp
from itertools import repeat

from scipy.stats import skew
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from ..utils.preprocessing_utils import create_baseline_datasets, \
        get_first, get_last
from ..utils.utils import get_subject_dirs


def parse_cl_args():
    """Parses CL arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-bp', '--baseline-data-path', type=str,
            default='data/baseline_features/',
            help='Path to the baseline data sets.')
    parser.add_argument('-sp', '--subjects-path', type=str, default='data/',
            help='Path to subject directories.')
    parser.add_argument('-v', '--verbose', type=str, default=True,
            help='Console output verbosity flag.')

    return parser.parse_args(argv[1:])


def main(args):
    if not os.path.exists(args.baseline_data_path):
        os.makedirs(args.baseline_data_path)
        train_dirs = get_subject_dirs(os.path.join(args.subjects_path,
            'train/'))
        test_dirs = get_subject_dirs(os.path.join(args.subjects_path, 'test/'))

        scaler = StandardScaler()

        with open('nicu_los/config.json') as f:
            config = json.load(f)
            variables = config['variables']
            sub_seqs = config['baseline_subsequences']

        stat_fns = [get_first, get_last, np.min, np.max, np.mean, np.std, skew,
                len]

        # Get training features and targets
        X_train, y_train, t_train = create_baseline_datasets(
                train_dirs, variables, stat_fns, sub_seqs)

        # Normalize X_train
        X_train = scaler.fit_transform(X_train)

        np.save('data/baseline_features/X_train', X_train)
        np.save('data/baseline_features/y_train', y_train)
        np.save('data/baseline_features/t_train', t_train)

        # Get test features and targets
        X_test, y_test, t_test  = create_baseline_datasets(
                test_dirs, variables, stat_fns, sub_seqs)

        # Normalize X_test
        X_test = scaler.transform(X_test)

        np.save('data/baseline_features/X_test', X_test)
        np.save('data/baseline_features/y_test', y_test)
        np.save('data/baseline_features/t_test', t_test)


if __name__ == '__main__':
    main(parse_cl_args())

