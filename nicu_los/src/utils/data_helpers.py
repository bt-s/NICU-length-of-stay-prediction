#!/usr/bin/python3

"""data_helpers.py

Various utility functions for data loading 
"""

__author__ = "Bas Straathof"

import json, os, random
import numpy as np
import pandas as pd

from sklearn.utils import shuffle
from tqdm import tqdm

from nicu_los.src.utils.utils import get_subject_dirs


def create_list_file(subject_dirs, list_file_path,
        ts_fname='timeseries_normalized.csv'):
    """Create a file containing a list of paths to timeseries data frames

    Args:
        subject_dirs (list): List of subject directories
        list_file_path (str): Path to the list file
        ts_fname (str): Name of the timeseries file
    """
    with open(list_file_path, 'a') as f:
        for i, sd in enumerate(tqdm(subject_dirs)):
            ts = pd.read_csv(os.path.join(sd, ts_fname))
            # Start from 4, since we only start predicting from the first four
            # hours of the stay
            for row in range(4, len(ts)+1):
                f.write(f'{sd}, {row}\n')


def data_generator(list_file, config='nicu_los/config.json',
        ts_file='timeseries_normalized.csv', coarse_targets=False,
        gestational_age=True, mask=True, task='classification',
        shuffle=True):
    """Data loader function

    Args:
        list_file (str): Path to the .txt file containing a list of (filename,
                         sequence length) combinations
        config (str): Path to the nicu_los config file
        ts_file (str): Name of the files containing the timeseries
        coarse_targets (bool): Whether to use coarse targets
        gestational_age (bool): Whether to use the gestational age variable 
        mask (bool): Whether to use missingness indicator variables
        task (str): One of 'classification'  and 'regression'
        shuffle (bool): Whether to shuffle the data

    Yields:
        X (np.ndarray): Features corresponding to one data batch

        EITHER (if task == 'regression'):
            y ():
        OR (if task == 'classification'):
            t ():
    """

    with open(list_file, 'r') as f:
        data = f.readlines()
    data = [line.split(',') for line in data]
    data = [(subject_dir, int(row)) for (subject_dir, row) in data]

    with open(config) as f:
        config = json.load(f)

    variables = config['variables']

    if not gestational_age and "GESTATIONAL_AGE_DAYS" in variables:
        variables.remove("GESTATIONAL_AGE_DAYS")

    if mask:
        variables = variables + ['mask_' + v for v in variables]
    
    if shuffle:
        random.shuffle(data)

    while True:
        if shuffle:
            random.shuffle(data)

        index = 0
        while index < len(data)-1:
            sd, row = data[index][0], data[index][1]
            index += 1
            ts = pd.read_csv(os.path.join(sd, ts_file))[:row]
            X = ts[variables].to_numpy()
            y = ts.LOS_HOURS.iloc[-1]

            if coarse_targets:
                t = ts.TARGET_COARSE.iloc[-1]
            else:
                t = ts.TARGET_FINE.iloc[-1]

            if task == 'regression':
                yield (X, y)
            else:
                yield (X, t)


def get_baseline_datasets(subject_dirs, coarse_targets=False, pre_imputed=False,
        targets_only=False, config='nicu_los/config.json'):
    """Obtain baseline data sets

    Args:
        subject_dirs (list): List of subject directories
        coarse_targets (bool): Whether to use coarse targets
        pre_imputed (bool): Whether to use features from pre-imputed data
        targets_only (bool): Whether to only load the targets

    Returns:
        X (np.ndarray|None): Features
        y (np.array): Targets -- remaining LOS
        t (np.array): Target -- buckets
    """
    tot_num_sub_seqs = 0
    for i, sd in enumerate(tqdm(subject_dirs)):
        tot_num_sub_seqs += len(pd.read_csv(os.path.join(sd,
            'timeseries.csv')))

    with open(config) as f:
        config = json.load(f)
        variables = config['variables']
        sub_seqs = config['baseline_subsequences']
        stat_fns = config['stat_fns']

        # Add the masks
        variables = ['mask_' + v for v in variables]

    if not targets_only:
        X = np.zeros((tot_num_sub_seqs,
            len(variables)*len(sub_seqs)*len(stat_fns)))
    else:
        X = None

    y, t = np.zeros(tot_num_sub_seqs), np.zeros(tot_num_sub_seqs)

    if coarse_targets:
        target_str = 'coarse'
    else:
        target_str = 'fine'

    pi_str = ''
    if pre_imputed:
        pi_str = '_pre_imputed'

    cnt = 0
    for i, sd in enumerate(tqdm(subject_dirs)):
        cnt_old = cnt
        if not targets_only:
            x = np.load(os.path.join(sd, f'X_baseline{pi_str}.npy'))

        yy = np.load(os.path.join(sd, f'y_baseline{pi_str}.npy'))
        tt = np.load(os.path.join(sd, f't_baseline_{target_str}{pi_str}.npy'))

        cnt += len(yy)

        if not targets_only:
            X[cnt_old:cnt, :] = x

        y[cnt_old:cnt] = yy
        t[cnt_old:cnt] = tt

    if not targets_only:
        X, y, t = shuffle(X, y, t)
        return X, y, t
    else:
        y, t = shuffle(y, t)
        return y, t


def get_optimal_bucket_boundaries(n=100):
    """Function to get the optimal bucket boundaries
    
    Args:
        n (int): Number of buckets

    Returns:
        bucket_boundaries (list): Optimal bucket boundaries for n
    """
    train_list_file = os.path.join('data', 'train_list.txt')
    val_list_file = os.path.join('data', 'val_list.txt')
    test_list_file = os.path.join('data', 'test_list.txt')
    list_files = [train_list_file, val_list_file, test_list_file]

    data = []
    for list_file in list_files :
        with open(list_file, 'r') as f:
            data += f.readlines()

    data = [line.split(',') for line in data]
    data = [(subject_dir, int(row)) for (subject_dir, row) in data]

    rows = []
    for _, r in data:
        rows.append(r)

    bucket_boundaries = []
    for i in range(100):
        bucket_boundaries.append(rows[len(rows)//100*i])

    return bucket_boundaries

