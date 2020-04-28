#!/usr/bin/python3

"""modelling_utils.py

Various utility functions for modelling
"""

__author__ = "Bas Straathof"

import errno, json, os

import pandas as pd
import numpy as np

from tqdm import tqdm

from ..utils.utils import get_subject_dirs

def get_baseline_datasets(subject_dirs):
    tot_num_sub_seqs = 0
    for i, sd in enumerate(tqdm(subject_dirs)):
        tot_num_sub_seqs += len(pd.read_csv(os.path.join(sd,
            'timeseries.csv')))

    X = np.zeros((tot_num_sub_seqs, 756))
    y, t = np.zeros(tot_num_sub_seqs), np.zeros(tot_num_sub_seqs)

    cnt = 0
    for i, sd in enumerate(tqdm(subject_dirs)):
        cnt_old = cnt
        x = np.load(os.path.join(sd,'X_baseline.npy'))
        yy = np.load(os.path.join(sd,'y_baseline.npy'))
        tt = np.load(os.path.join(sd,'t_baseline.npy'))

        cnt += len(yy)

        X[cnt_old:cnt, :] = x
        y[cnt_old:cnt] = yy
        t[cnt_old:cnt] = tt

    return X, y, t

def get_train_val_test_baseline_sets(data_path, task):
    with open(os.path.join(data_path, 'training_subjects.txt'), 'r') as f:
        train_dirs = f.read().splitlines()
    with open(os.path.join(data_path, 'validation_subjects.txt') , 'r') as f:
        val_dirs = f.read().splitlines()
    with open(os.path.join(data_path, 'test_subjects.txt'), 'r') as f:
        test_dirs = f.read().splitlines()

    X_train = np.load(os.path.join(data_path,'X_train.npy'))
    y_train = np.load(os.path.join(data_path, 'y_train.npy'))
    t_train = np.load(os.path.join(data_path, 't_train.npy'))
    X_val = np.load(os.path.join(data_path,'X_val.npy'))
    y_val = np.load(os.path.join(data_path, 'y_val.npy'))
    t_val = np.load(os.path.join(data_path, 't_val.npy'))
    X_test = np.load(os.path.join(data_path,'X_test.npy'))
    y_test = np.load(os.path.join(data_path, 'y_test.npy'))
    t_test = np.load(os.path.join(data_path, 't_test.npy'))

    if task == 'classification':
        return X_train, t_train, X_val, t_val, X_test, t_test
    elif task == 'regression':
        return X_train, y_train, X_val, y_val, X_test, y_test
    else:
        raise ValueError("Task must be one of: 'classification', 'regression'.")


def get_train_val_test_data(data_path, task):
    train_subject_directories = get_subject_dirs(os.path.join(data_path,
        'train/'))
    test_subject_directories = get_subject_dirs(os.path.join(data_path,
        'test/'))

    with open('nicu_los/config.json') as f:
        config = json.load(f)
        variables = config['variables']

        # Add the masks
        variables = ['mask_' + v for v in variables]

    for i, sd in enumerate(train_subject_directories):
        # Read the normalized and inputed timeserie
        df_ts = pd.read_csv(os.path.join(sd, 'timeseries_normalized.csv'))

        t = df_ts.TARGET.to_list()
        y = df_ts.LOS_HOURS.to_list()
        X = df_ts[variables].to_numpy()
        print(X.shape)
        exit()

    if task == 'classification':
        return X_train, t_train, X_val, t_val, X_test, t_test
    elif task == 'regression':
        return X_train, y_train, X_val, y_val, X_test, y_test
    else:
        raise ValueError("Task must be one of: 'classification', 'regression'.")


