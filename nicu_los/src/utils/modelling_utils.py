#!/usr/bin/python3

"""modelling_utils.py

Various utility functions for modelling
"""

__author__ = "Bas Straathof"

import errno, json, os, random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.metrics import accuracy_score, cohen_kappa_score, \
        confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, \
        plot_confusion_matrix, precision_score, recall_score, roc_auc_score

from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, GRU, Input, \
        LSTM, Masking
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

from nicu_los.src.utils.utils import get_subject_dirs


def get_baseline_datasets(subject_dirs, coarse_targets=False):
    """Obtain baseline data sets

    Args:
        subject_dirs (list): List of subject directories
        coarse_targets (bool): Whether to use coarse targets

    Returns:
        X (np.ndarray): Features
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

    X = np.zeros((tot_num_sub_seqs, len(variables)*len(sub_seqs)*len(stat_fns)))
    y, t = np.zeros(tot_num_sub_seqs), np.zeros(tot_num_sub_seqs)

    if coarse_targets:
        target_str = 'coarse'
    else:
        target_str = 'fine'

    cnt = 0
    for i, sd in enumerate(tqdm(subject_dirs)):
        cnt_old = cnt
        x = np.load(os.path.join(sd, f'X_baseline.npy'))
        yy = np.load(os.path.join(sd,f'y_baseline.npy'))
        tt = np.load(os.path.join(sd,f't_baseline_{target_str}.npy'))

        cnt += len(yy)

        X[cnt_old:cnt, :] = x
        y[cnt_old:cnt] = yy
        t[cnt_old:cnt] = tt

    return X, y, t


def get_train_val_test_baseline_sets(data_path, task):
    """Get the training, validation and test baseline data sets

    Args:
        data_path (str): Path to the data directory
        task (str): One of 'regression' and 'classification'

    Returns:
        X_train (np.ndarray): Training features
        X_val (np.ndarray): Validation features
        X_test (np.ndarray): Test features

        EITHER (if task == 'regression'):
            y_train (np.ndarray): Training targets -- remaining LOS
            y_val (np.ndarray): Validation targets -- remaining LOS
            y_test (np.ndarray): Test targets -- remaining LOS

        OR (if task == 'classification'):
            t_train (np.ndarray): Training targets -- buckets
            t_val (np.ndarray): Validation targets -- buckets
            t_test (np.ndarray): Test targets -- buckets
    """
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
        coarse_targets (bool): Whether to use coarse targets
        mask (bool): Whether to use missingness indicator variables
    """
    def __init__(self, list_file, config='nicu_los/config.json',
            ts_file='timeseries_normalized.csv', coarse_targets=False,
            mask=True):
        self.current_index = 0
        self.ts_file = ts_file
        self.coarse_targets = coarse_targets

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

        if self.coarse_targets:
            t = ts.TARGET_COARSE.iloc[-1]
        else:
            t = ts.TARGET_FINE.iloc[-1]

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


def construct_rnn(model_type='lstm', n_cells=1, input_dimension=28,
        dropout=0.3, hid_dimension=64, model_name=""):
    """Construct an RNN model (either LSTM or GRU)

    Args:
        n_cells (int): Number of RNN cells
        input_dimension (int): Input dimension of the model
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

    y = Dense(units=10, activation='softmax')(X)
    outputs = [y]

    return Model(inputs=inputs, outputs=outputs, name=model_name)


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

            (Xs, ys, ts) = sort_and_batch_shuffle(data, batch_size)

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
    dropout = model_params['dropout']
    hid_dimension = model_params['hidden_dimension']

    model = construct_rnn(model_type, n_cells, input_dimension, dropout,
            hid_dimension, model_name)

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
        print('\nPredict on training data:\n')
        y_true, y_pred = [], []
        for batch, (x, y) in enumerate(self.training_data):
            if batch > self.training_steps:
                break

            y_pred.append(np.argmax(self.model.predict_on_batch(x), axis=1))
            y_true.append(y.numpy())

        evaluate_classification_model(np.concatenate(y_true, axis=0),
                np.concatenate(y_pred, axis=0))

        print('\nPredict on validation data:\n')
        y_true, y_pred = [], []
        for batch, (x, y) in enumerate(self.validation_data):
            if batch > self.validation_steps:
                break

            y_pred.append(np.argmax(self.model.predict_on_batch(x), axis=1))
            y_true.append(y.numpy())

        evaluate_classification_model(np.concatenate(y_true, axis=0),
                np.concatenate(y_pred, axis=0))


def mean_absolute_perc_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 0.1))) * 100


def evaluate_classification_model(y_true, y_pred, verbose=1):
    kappa = cohen_kappa_score(y_true, y_pred, weights='linear')
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)

    if verbose:
        print(f'Accuracy: {acc}')
        print(f'Linear Cohen Kappa Score: {kappa}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'Confusion matrix:\n{cm}')

    return {"accuracy": acc, 'kappa': kappa, 'precision': precision,
            'recall': recall, 'cm': cm}


def evaluate_regression_model(y_true, y_pred, verbose=1):
    mad = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_perc_error(y_true, y_pred)

    if verbose:
        print(f'Mean Absolute Deviation (MAD): {mad}')
        print(f'Mean Squared Error (MSE): {mse}')
        print(f'Root Mean Squared Error (RMSE): {rmse}')
        print(f'Mean Aboslute Perentage Error (MAPE): {mape}')

    return {'mad': mad, 'mse': mse, 'rmse': rmse, 'mape': mape}


def get_confusion_matrix(model, X, y, save_plot='', class_names=['0-1',
    '1-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8', '8-14', '14+']):
    titles_options = [("Confusion matrix, without normalization", None),
                    ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(model, X, y,
                                    display_labels=class_names,
                                    cmap=plt.cm.Blues,
                                    normalize=normalize)
        disp.ax_.set_title(title)

        if save_plot:
            if normalize: save_plot += '_normalized'
            plt.savefig(save_plot, format="pdf", bbox_inches='tight',
                    pad_inches=0)
            plt.close()
        else:
            plt.show()
            plt.close()

