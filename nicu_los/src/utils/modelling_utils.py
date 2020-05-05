#!/usr/bin/python3

"""modelling_utils.py

Various utility functions for modelling
"""

__author__ = "Bas Straathof"

import errno, json, os, random

import pandas as pd
import numpy as np

from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM

from ..utils.utils import get_subject_dirs


def get_baseline_datasets(subject_dirs):
    tot_num_sub_seqs = 0
    for i, sd in enumerate(tqdm(subject_dirs)):
        tot_num_sub_seqs += len(pd.read_csv(os.path.join(sd,
            'timeseries.csv')))

    with open('nicu_los/config.json') as f:
        config = json.load(f)
        variables = config['variables']
        sub_seqs = config['baseline_subsequences']
        stat_fns = config['stat_fns']

        # Add the masks
        variables = ['mask_' + v for v in variables]

    X = np.zeros((tot_num_sub_seqs, len(variables)*len(sub_seqs)*len(stat_fns)))
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
        raise ValueError("Task must be one of: 'classification', \
                'regression'.")



class TimeSeriesReader(object):
    """Reader to read length-of-stay timeseries sequences from a list file

    Attributes:
        list_file (str): Path to the .txt file containing a list of (filename,
                         sequence length) combinations
        config (str): Path to the nicu_los config file
        ts_file (str): Name of the files containing the timeseries
        mask (bool): Whether to use missingness indicator variables
    """
    def __init__(self, list_file, config='nicu_los/config.json',
            ts_file='timeseries_normalized.csv', mask=True):
        self.current_index = 0
        self.ts_file = ts_file

        with open(list_file, 'r') as f:
            self.data = f.readlines()

        self.data = [line.split(',') for line in self.data]
        self.data = [(subject_dir, int(row)) for (subject_dir, row) in \
                self.data]

        with open(config) as f:
            config = json.load(f)
            self.variables = config['variables']
            if mask:
                self.variables = self.variables + \
                        ['mask_' + v for v in self.variables]

    def random_shuffle(self, seed=None):
        if seed: random.seed(seed)
        random.shuffle(self.data)

    def get_number_of_sequences(self):
        return len(self.data)

    def read_sequence(self, index):
        """Read a timeseries sequence

        Args:
            index (int): Index of the reader (i.e. index of the list file)

        Returns:
            (dict): Dictionary containing the training features, the target
                    y (i.e. remaining length-of-stay in # of hours) and the
                    target t (i.e. the bucket corresponding to y)
        """
        if index < 0 or index >= len(self.data):
            raise ValueError('Invalid index.')

        sd, row = self.data[index][0], self.data[index][1]

        ts = pd.read_csv(os.path.join(sd, self.ts_file))[:row]

        X = ts[self.variables].to_numpy()
        y = ts.LOS_HOURS.iloc[-1]
        t = ts.TARGET.iloc[-1]

        return {'X': X, 'y': y, 't': t}

    def read_chunk(self, chunk_size):
        """Read a specific number of timeseries sequences

        Args:
            chunk_size (int): Size of the chunk

        Returns:
            data (dict): Dictionary containing a list of training features,
                         a list of the targets y and a list of the target t
        """
        data = {}
        for i in range(chunk_size):
            for k, v in self.read_next().items():
                if k not in data:
                    data[k] = []
                data[k].append(v)

        return data

    def read_next(self):
        self.current_index += 1

        if self.current_index == self.get_number_of_sequences():
            self.current_index = 0

        return self.read_sequence(self.current_index)


def construct_simple_lstm(input_dimension=28, dropout=0.3, hid_dimension=64):
    X = Input(shape=(None, input_dimension))
    inputs = [X]

    num_hid_units = hid_dimension

    X = LSTM(activation='tanh', dropout=dropout,
            recurrent_dropout=dropout,
            return_sequences=False,
            units=num_hid_units)(inputs)

    if dropout > 0:
        X = Dropout(dropout)(X)

    y = Dense(units=10, activation='softmax')(X)
    outputs = [y]

    return Model(inputs=inputs, outputs=outputs, name='simple_lstm')


def sort_and_shuffle(data, batch_size):
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
    batches = [data[i: i + batch_size] for i in range(0, len(data), batch_size)]

    # Shuffle the batches randomly
    random.shuffle(batches)

    # Flatten the batches and zip the data together
    data = [x for batch in batches for x in batch]
    data = list(zip(*data))

    return data


def zero_pad_timeseries(batch):
    dtype = batch[0].dtype
    max_len = max([x.shape[0] for x in batch])

    padded_batch = np.array(
            [np.concatenate(
                [x, np.zeros((max_len - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
                for x in batch])

    return padded_batch


def data_generator(list_file, steps, batch_size, task='classification',
        mask=True, shuffle=True):
    reader = TimeSeriesReader(list_file, mask=mask)

    if steps:
        chunk_size = steps*batch_size
    else:
        chunk_size = reader.get_number_of_sequences()
        cnt = 0

    while True:
        if shuffle and reader.current_index == 0:
            reader.random_shuffle(seed=42)

        remaining = chunk_size
        if not steps:
            if cnt == 1:
                break
            else:
                cnt +=1

        while remaining > 0:
            current_size = min(chunk_size, remaining)
            remaining -= current_size

            data = reader.read_chunk(current_size)

            (Xs, ys, ts) = sort_and_shuffle(data, batch_size)

            for i in range(0, current_size, batch_size):
                X = zero_pad_timeseries(Xs[i:i + batch_size])
                y = ys[i:i+batch_size]
                t = ts[i:i+batch_size]

                if task == 'regression':
                    yield X, y
                else:
                    yield X, t


def create_list_file(subject_dirs, list_file_path,
        ts_fname='timeseries_normalized.csv'):
    with open(list_file_path, 'a') as f:
        for i, sd in enumerate(tqdm(subject_dirs)):
            ts = pd.read_csv(os.path.join(sd, ts_fname))
            for row in range(1, len(ts)+1):
                f.write(f'{sd}, {row}\n')
