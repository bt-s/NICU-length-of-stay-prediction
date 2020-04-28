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
from sklearn.impute import SimpleImputer

from ..utils.preprocessing_utils import create_baseline_datasets, \
        get_first, get_last, split_train_val
from ..utils.utils import get_subject_dirs


def parse_cl_args():
    """Parses CL arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-bp', '--baseline-data-path', type=str,
            default='data/baseline_features/',
            help='Path to the baseline data sets.')
    parser.add_argument('-sp', '--subjects-path', type=str, default='data/',
            help='Path to subject directories.')
    parser.add_argument('-pi', '--pre-imputed', type=int, default=0,
            help='Whether to use pre-imputed time-series.')
    parser.add_argument('-v', '--verbose', type=str, default=True,
            help='Console output verbosity flag.')

    return parser.parse_args(argv[1:])


def main(args):
    baseline_data_path = args.baseline_data_path
    subjects_path = args.subjects_path
    pre_imputed = args.pre_imputed

    if not os.path.exists(baseline_data_path):
        os.makedirs(baseline_data_path)

    train_sub_path = os.path.join(baseline_data_path, 'training_subjects.txt')
    val_sub_path = os.path.join(baseline_data_path,
            'validation_subjects.txt')
    test_sub_path = os.path.join(baseline_data_path, 'test_subjects.txt')

    if os.path.exists(train_sub_path) and os.path.exists(val_sub_path) \
            and os.path.exists(test_sub_path):
        with open(train_sub_path, 'r') as f:
            train_dirs = f.read().splitlines()
        with open(val_sub_path, 'r') as f:
            val_dirs = f.read().splitlines()
        with open(test_sub_path, 'r') as f:
            test_dirs = f.read().splitlines()
    else:
        test_dirs = get_subject_dirs(os.path.join(subjects_path, 'test/'))

        # Split the training directories into 80% training directories and
        # 20% validation directories
        train_dirs, val_dirs = split_train_val(os.path.join(subjects_path,
            'train/'), val_perc=0.2)

        # Write the training, validation and test subjects to files
        with open(train_sub_path,'w') as f: f.write('\n'.join(train_dirs))
        with open(val_sub_path,'w') as f: f.write('\n'.join(val_dirs))
        with open(test_sub_path,'w') as f: f.write('\n'.join(test_dirs))


    with open('../../config.json') as f:
        config = json.load(f)
        variables = config['variables']
        sub_seqs = config['baseline_subsequences']

    # Functions to compute statistical features
    stat_fns = [np.min, np.max, np.mean, np.std, skew, len]

    # Get training features and targets
    X_train, y_train, t_train = create_baseline_datasets(train_dirs,
            variables, stat_fns, sub_seqs, pre_imputed)

    if not pre_imputed:
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean',
                fill_value='constant', verbose=0, copy=True)
        imputer.fit(X_train)
        X_train = imputer.transform(X_train)

    # Normalize X_train
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    test = 0
    np.save('data/baseline_features/X_train', X_train)
    np.save('data/baseline_features/y_train', y_train)
    np.save('data/baseline_features/t_train', t_train)

    # Get validation features and targets
    X_val, y_val, t_val  = create_baseline_datasets(val_dirs, variables,
            stat_fns, sub_seqs, pre_imputed)

    if not pre_imputed:
        # Impute X_val
        X_val = imputer.transform(X_val)

    # Normalize X_val
    X_val = scaler.transform(X_val)

    np.save('data/baseline_features/X_val', X_val)
    np.save('data/baseline_features/y_val', y_val)
    np.save('data/baseline_features/t_val', t_val)

    # Get test features and targets
    X_test, y_test, t_test  = create_baseline_datasets(test_dirs,
            variables, stat_fns, sub_seqs, pre_imputed)

    if not pre_imputed:
        # Impute X_test
        X_test = imputer.transform(X_test)

    # Normalize X_test
    X_test = scaler.transform(X_test)

    np.save('data/baseline_features/X_test', X_test)
    np.save('data/baseline_features/y_test', y_test)
    np.save('data/baseline_features/t_test', t_test)


if __name__ == '__main__':
    main(parse_cl_args())

