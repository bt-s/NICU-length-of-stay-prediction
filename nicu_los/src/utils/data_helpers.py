#!/usr/bin/python3

"""data_helpers.py

Various utility functions for data loading 
"""

__author__ = "Bas Straathof"

import json, os, random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils import shuffle
from tqdm import tqdm

from nicu_los.src.utils.utils import get_subject_dirs
from nicu_los.src.utils.readers import TimeSeriesReader


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


def data_generator(reader, steps, batch_size, task, shuffle=True):
    """Data loader function

    Args:
        reader (TimeSeriesReader): Time series reader object
        steps (int): Number of steps per epoch
        batch_size (int): Training batch size
        taks (str): One of 'classification'  and 'regression'
        shuffle (bool): Whether to shuffle the data

    Yields:
        X (np.ndarray): Features corresponding to one data batch

        EITHER (if task == 'regression'):
            y ():
        OR (if task == 'classification'):
            t ():
    """
    n_examples = reader.get_number_of_sequences()
    if not steps:
        steps = (n_examples + batch_size - 1) // batch_size
        n_examples_epoch = 2048 
        print(f"\n==> {reader.name} -- number of examples:",
                n_examples)
    else:
        n_examples_epoch = steps * batch_size
        print(f"\n==> {reader.name} -- number of examples per epoch:",
                n_examples_epoch)

    # Set a limit on the size of the chunk to be read
    chunk_size = min(2048, steps) * batch_size

    while True:
        # Shuffle once per training round
        if shuffle:
            if (reader.current_index == 0) or (reader.current_index >
                batch_size*steps):
                reader.random_shuffle()
                reader.current_index = 0 

        n_examples_remaining = n_examples_epoch
        while n_examples_remaining > 0:
            current_size = min(chunk_size, n_examples_remaining)
            n_examples_remaining -= current_size

            data = reader.read_chunk(current_size)

            if batch_size > 1:
                (Xs, ys, ts) = sort_and_batch_shuffle(data, batch_size)
            else:
                Xs, ys, ts = data['X'], data['y'], data['t']

            for i in range(0, current_size, batch_size):
                X = zero_pad_timeseries(Xs[i:i + batch_size])
                y = ys[i:i+batch_size]
                t = ts[i:i+batch_size]

                if task == 'regression':
                    yield X, y
                else:
                    yield X, t


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

    X, y, t = shuffle(X, y, t)

    return X, y, t


def sort_and_batch_shuffle(data, batch_size):
    """Sort the data set and shuffle the batches

    Args:
        data ():
        batch_size (int):

    Returns:
        data ():
    """
    # Unpack Xs, ys, ts
    Xs = data['X']
    ys = data['y']
    ts = data['t']

    # Zip the data together
    data = list(zip(Xs, ys, ts))

    # Find and drop the remainder
    remainder = len(data) % batch_size
    data = data[:len(data)- remainder]

    # Sort the data by length of the time series
    data.sort(key=(lambda x: x[0].shape[0]))

    # Create batches of time series that are closest in length
    batches = [data[i: i + batch_size] for i in range(0, len(data),
        batch_size)]

    # Shuffle the batches randomly
    random.shuffle(batches)

    # Flatten the batches and zip the data together
    data = [x for batch in batches for x in batch]
    data = list(zip(*data))

    return data


def zero_pad_timeseries(batch):
    """Zero pads a batch of time series to the length of of the longest
       series in the batch

    Args:
        batch (list): Input batch (batch_size, None, channels)

    Returns:
        batch (list): Padded batch (batch_size, longest_time_dim, channels)
    """
    batch = pad_sequences(batch, padding='post')

    return batch

