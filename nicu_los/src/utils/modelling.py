#!/usr/bin/python3

"""modelling.py

Various utility functions for modelling
"""

__author__ = "Bas Straathof"

import errno, json, os, random

import numpy as np
import pandas as pd

from tqdm import tqdm

from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, GRU, Input, \
        LSTM, Masking
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

from nicu_los.src.utils.utils import get_subject_dirs
from nicu_los.src.utils.readers import TimeSeriesReader 
from nicu_los.src.utils.evaluation import evaluate_classification_model


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
            for row in range(1, len(ts)+1):
                f.write(f'{sd}, {row}\n')


def data_generator(list_file, steps, batch_size, task='classification',
        coarse_targets=False, mask=True, shuffle=True):
    """Data loader function

    Args:
        list_file (str): Path to a file that contains a list of paths to
                         timeseries
        steps (int): Number of steps per epoch
        batch_size (int): Training batch size
        taks (str): One of 'classification'  and 'regression'
        coarse_targets (bool): Whether to use coarse targets
        mask (bool): Whether to mask the variables
        shuffle (bool): Whether to shuffle the data

    Yields:
        X (np.ndarray): Features corresponding to one data batch

        EITHER (if task == 'regression'):
            y ():
        OR (if task == 'classification'):
            t ():
    """
    reader = TimeSeriesReader(list_file, coarse_targets=coarse_targets,
            mask=mask)

    chunk_size = steps*batch_size

    while True:
        if shuffle and reader.current_index == 0:
            reader.random_shuffle(seed=42)

        remaining = chunk_size

        while remaining > 0:
            current_size = min(chunk_size, remaining)
            remaining -= current_size

            data = reader.read_chunk(current_size)

            (Xs, ys, ts) = sort_and_batch_shuffle(data, batch_size)

            for i in range(0, current_size, batch_size):
                X = zero_pad_timeseries(Xs[i:i + batch_size])
                y = ys[i:i+batch_size]
                t = ts[i:i+batch_size]

                if task == 'regression':
                    yield X, y
                else:
                    yield X, t


def get_baseline_datasets(subject_dirs, coarse_targets=False,
        pre_imputed=False, targets_only=False):
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

    with open('nicu_los/config.json') as f:
        config = json.load(f)
        variables = config['variables']
        sub_seqs = config['baseline_subsequences']
        stat_fns = config['stat_fns']

        # Add the masks
        variables = ['mask_' + v for v in variables]

    if not targets_only:
        X = np.zeros((tot_num_sub_seqs, len(variables)*len(sub_seqs)*len(stat_fns)))
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
    batches = [data[i: i + batch_size] for i in range(0, len(data), batch_size)]

    # Shuffle the batches randomly
    random.shuffle(batches)

    # Flatten the batches and zip the data together
    data = [x for batch in batches for x in batch]
    data = list(zip(*data))

    return data


def zero_pad_timeseries(batch):
    """Zero pad a timeseries batch to the length of of the longest series

    Args:
        batch ():

    Returns:
        batch ():
    """
    dtype = batch[0].dtype
    max_len = max([x.shape[0] for x in batch])

    # Pad the batch
    batch = [np.concatenate([x, np.zeros((max_len - x.shape[0],) + x.shape[1:],
        dtype=dtype)], axis=0) for x in batch]
    batch = np.array(batch)

    return batch


def construct_rnn(input_dimension, output_dimension, model_type='lstm',
        n_cells=1, dropout=0.3, hid_dimension=64, model_name=""):
    """Construct an RNN model (either LSTM or GRU)

    Args:
        input_dimension (int): Input dimension of the model
        output_dimension (int): Output dimension of the model
        n_cells (int): Number of RNN cells
        dropout (float): Amount of dropout to apply
        hid_dimension (int): Dimension of the hidden layer (i.e. # of unit in
                             the RNN cell)

    Returns:
        model (tf.keras.Model): Constructed RNN model
    """
    X = Input(shape=(None, input_dimension))
    inputs = [X]

    # Skip timestep if all  values of the input tensor are 0
    X = Masking()(X)

    num_hid_units = hid_dimension

    for layer in range(n_cells - 1):
        num_hid_units = num_hid_units // 2

        if model_type == 'lstm':
            cell = LSTM(units=num_hid_units, activation='tanh',
                    return_sequences=True, recurrent_dropout=dropout,
                    dropout=dropout)
        elif model_type == 'gru':
            cell = GRU(units=num_hid_units, activation='tanh',
                    return_sequences=True, recurrent_dropout=dropout,
                    dropout=dropout)
        else:
            raise ValueError("Parameter 'model_type' should be one of " +
                    "'lstm' or 'gru'.")

        X = Bidirectional(cell)(X)

    # There always has to be at least one cell
    if model_type == 'lstm':
        X = LSTM(activation='tanh', dropout=dropout, recurrent_dropout=dropout,
                return_sequences=False, units=hid_dimension)(X)
    elif model_type == 'gru':
        X = GRU(activation='tanh', dropout=dropout, recurrent_dropout=dropout,
                return_sequences=False, units=hid_dimension)(X)
    else:
        raise ValueError("Parameter 'model_type' should be one of " +
                "'lstm' or 'gru'.")

    if dropout:
        X = Dropout(dropout)(X)

    y = Dense(units=output_dimension, activation='softmax')(X)
    outputs = [y]

    return Model(inputs=inputs, outputs=outputs, name=model_name)


def construct_and_compile_model(model_type, model_name, checkpoint_file,
        checkpoints_dir, model_params={}):
    """Construct and compile a model of a specific type

    Args:
        model_type (str): The type of model to be constructed
        checkpoint_file (str): Name of a checkpoint file
        checkpoints_dir (str): Path to the checkpoints directory
        model_params (dict): Possible hyper-parameters for the model to be
                             constructed

    Returns:
        model (tf.keras.Model): Constructed and compiled model
    """
    n_cells = model_params['n_cells']
    input_dimension = model_params['input_dimension']
    output_dimension = model_params['output_dimension']
    dropout = model_params['dropout']
    hid_dimension = model_params['hidden_dimension']

    model = construct_rnn(input_dimension, output_dimension, model_type,
            n_cells, dropout, hid_dimension, model_name)

    if checkpoint_file:
        model.load_weights(os.path.join(checkpoints_dir, checkpoint_file))

    model.compile(
            optimizer=Adam(),
            loss=SparseCategoricalCrossentropy(),
            metrics=['accuracy'])

    return model


class MetricsCallback(Callback):
    def __init__(self, model, training_data, validation_data, training_steps,
            validation_steps):
        self.model = model
        self.training_data = training_data
        self.validation_data = validation_data

        self.training_steps = training_steps
        self.validation_steps = validation_steps

    def on_epoch_end(self, epoch, logs=None):
        print('\n=> Predict on training data:\n')
        y_true, y_pred = [], []
        for batch, (x, y) in enumerate(self.training_data):
            if batch > self.training_steps:
                break

            y_pred.append(np.argmax(self.model.predict_on_batch(x), axis=1))
            y_true.append(y.numpy())

        evaluate_classification_model(np.concatenate(y_true, axis=0),
                np.concatenate(y_pred, axis=0))

        print('\n=> Predict on validation data:\n')
        y_true, y_pred = [], []
        for batch, (x, y) in enumerate(self.validation_data):
            if batch > self.validation_steps:
                break

            y_pred.append(np.argmax(self.model.predict_on_batch(x), axis=1))
            y_true.append(y.numpy())

        evaluate_classification_model(np.concatenate(y_true, axis=0),
                np.concatenate(y_pred, axis=0))


