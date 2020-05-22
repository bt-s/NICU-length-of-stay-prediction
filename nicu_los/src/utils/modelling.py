#!/usr/bin/python3

"""modelling.py

Various utility functions for modelling
"""

__author__ = "Bas Straathof"

import errno, json, os, random

import numpy as np
import pandas as pd

from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, BatchNormalization, \
        Bidirectional, concatenate, Conv1D, Dense, Dropout, \
        GlobalAveragePooling1D, GRU, Input, Layer, LSTM, Masking, multiply, \
        Reshape
from tensorflow.keras.losses import MeanAbsoluteError, SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

from sklearn.utils import shuffle

from nicu_los.src.utils.utils import get_subject_dirs
from nicu_los.src.utils.readers import TimeSeriesReader
from nicu_los.src.utils.evaluation import evaluate_classification_model, \
        evaluate_regression_model


class Slice(Layer):
    """Take a channel (i.e. feature-mask combination) slice of a 3D tensor"""
    def __init__(self, feature, mask, **kwargs):
        self.slice = [feature, mask]
        super(Slice, self).__init__(**kwargs)

    def call(self, X, mask=None):
        # Permute the tensor
        X = tf.transpose(X, perm=(2, 0, 1))

        # Gather the feature and corresponding mask
        X = tf.gather(X, self.slice)

        # Re-permute the tensor
        X = tf.transpose(X, perm=(1, 2, 0))

        return X

    # Layer must override get_config
    def get_config(self):
        return {'slice': self.slice}


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
        n_examples_epoch = n_examples
    else:
        n_examples_epoch = steps * batch_size
    print(f"\n==> {reader.name} -- number of examples per epoch:",
            n_examples_epoch)

    # Set a limit on the size of the chunk to be read
    chunk_size = min(1024, steps) * batch_size

    while True:
        # Only shuffle when the current reader index is 0, or if we are close
        # to the end of the reader -- then we reset the index
        if shuffle and (reader.current_index == 0 or reader.current_index >
                (n_examples - chunk_size)):
            reader.random_shuffle()
            reader.current_index = 0 

        n_examples_remaining = n_examples_epoch
        while n_examples_remaining > 0:
            current_size = min(chunk_size, n_examples_remaining)
            n_examples_remaining -= current_size

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


def get_baseline_datasets(subject_dirs, coarse_targets=False, pre_imputed=False,
        targets_only=False):
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
    inputs  = Input(shape=(None, input_dimension))

    # Skip timestep if all  values of the input tensor are 0
    X = Masking()(inputs)

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

    if output_dimension != 1:
        # Classification
        outputs = Dense(units=output_dimension, activation='softmax')(X)
    else:
        # Regression 
        outputs = Dense(units=output_dimension)(X)

    model = Model(inputs=inputs, outputs=outputs, name=model_name)

    return model


def construct_fcn(input_dimension, output_dimension, hid_dimension_lstm=8,
        model_name=""):
    """Construct an FCN model for multivariate time series classification
    
    (Karim et al. 2019 - Multivariate LSTM-FCNs for time series classification)

    Args:
        input_dimension (int): Input dimension of the model
        output_dimension (int): Output dimension of the model
        hid_dimension (int): Dimension of the hidden layer (i.e. # of unit in
                             the RNN cell)
        model_name (str): Name of the model

    Returns:
        model (tf.keras.Model): Constructed CN model
    """

    inputs = Input(shape=(None, input_dimension))

    X = Conv1D(128, 8, padding='same',
           kernel_initializer='he_uniform')(inputs)
    X = BatchNormalization()(X2)
    X = Activation('relu')(X2)
    X = squeeze_excite_block(X2)

    X = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(X2)
    X = BatchNormalization()(X2)
    X = Activation('relu')(X2)
    X = squeeze_excite_block(X2)

    X = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(X2)
    X = BatchNormalization()(X2)
    X = Activation('relu')(X2)

    X = GlobalAveragePooling1D()(X2)

    if output_dimension != 1:
        # Classification
        outputs = Dense(units=output_dimension, activation='softmax')(X)
    else:
        # Regression 
        outputs = Dense(units=output_dimension)(X)

    model = Model(inputs=inputs, outputs=outputs, name=model_name)

    return model


def construct_lstm_fcn(input_dimension, output_dimension, dropout=0.8,
        hid_dimension_lstm=8, model_name=""):
    """Construct an LSTM-FCN model 
    
    (Karim et al. 2019 - Multivariate LSTM-FCNs for time series classification)

    Args:
        input_dimension (int): Input dimension of the model
        output_dimension (int): Output dimension of the model
        dropout (float): Amount of dropout to apply
        hid_dimension (int): Dimension of the hidden layer (i.e. # of unit in
                             the RNN cell)
        model_name (str): Name of the model

    Returns:
        model (tf.keras.Model): Constructed LSTM-FCN model
    """

    inputs = Input(shape=(None, input_dimension))

    X1 = Masking()(inputs)
    X1 = LSTM(hid_dimension_lstm)(X1)
    X1 = Dropout(dropout)(X1)

    X2 = Conv1D(128, 8, padding='same',
            kernel_initializer='he_uniform')(inputs)
    X2 = BatchNormalization()(X2)
    X2 = Activation('relu')(X2)
    X2 = squeeze_excite_block(X2)

    X2 = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(X2)
    X2 = BatchNormalization()(X2)
    X2 = Activation('relu')(X2)
    X2 = squeeze_excite_block(X2)

    X2 = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(X2)
    X2 = BatchNormalization()(X2)
    X2 = Activation('relu')(X2)

    X2 = GlobalAveragePooling1D()(X2)

    X = concatenate([X1, X2])

    if output_dimension != 1:
        # Classification
        outputs = Dense(units=output_dimension, activation='softmax')(X)
    else:
        # Regression 
        outputs = Dense(units=output_dimension)(X)

    model = Model(inputs=inputs, outputs=outputs, name=model_name)

    return model


def construct_channel_wise_rnn(input_dimension, output_dimension,
        model_type='lstm_cw', dropout=0.0, global_dropout=0.0,
        hid_dimension=16, multiplier=4, model_name=""):
    """Construct an RNN model (either LSTM or GRU)

    Args:
        input_dimension (int): Input dimension of the model
        output_dimension (int): Output dimension of the model
        dropout (float): Amount of dropout to apply
        global_dropout (float): Amount of dropout to apply to the global dropout layer
        hid_dimension (int): Dimension of the hidden layer (i.e. # of unit in
                             the RNN cell)
        multiplier (int): Multiplier for the hidden dimension of the global LSTM 

    Returns:
        model (tf.keras.Model): Constructed channel-wise RNN model
    """
    inputs = Input(shape=(None, input_dimension))

    # Skip timestep if all  values of the input tensor are 0
    X = Masking()(inputs)

    # Train LSTMs over the channels, and append them
    cXs = []
    for feature in range(int(input_dimension/2)):
        mask = int(feature+input_dimension/2)
        channel_slice = Slice(feature, mask)(X)

        num_hid_units = hid_dimension // 2

        cell = LSTM(units=num_hid_units, activation='tanh',
                return_sequences=True, recurrent_dropout=dropout,
                dropout=dropout)

        cXs.append(Bidirectional(cell)(channel_slice))

    # Concatenate the channels
    X = concatenate(cXs, axis=2)

    # There always has to be at least one cell
    if model_type == 'lstm_cw':
        X = LSTM(activation='tanh', dropout=dropout, recurrent_dropout=0.2,
                return_sequences=False, units=multiplier*hid_dimension)(X)
    elif model_type == 'gru_cw':
        X = GRU(activation='tanh', dropout=dropout, recurrent_dropout=0.2,
                return_sequences=False, units=multiplier*hid_dimension)(X)
    else:
        raise ValueError("Parameter 'model_type' should be one of " +
                "'lstm_cw' or 'gru_cw'.")

    if global_dropout:
        X = Dropout(global_dropout)(X)

    if output_dimension != 1:
        # Classification
        outputs = Dense(units=output_dimension, activation='softmax')(X)
    else:
        # Regression 
        outputs = Dense(units=output_dimension)(X)

    model = Model(inputs=inputs, outputs=outputs, name=model_name)

    return model


def construct_and_compile_model(model_type, model_name, task, checkpoint_file,
        checkpoints_dir, model_params={}):
    """Construct and compile a model of a specific type

    Args:
        model_type (str): The type of model to be constructed
        model_name (str): The name of model to be constructed
        task (str): Either 'regression' or 'classification'
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
    global_dropout = model_params['global_dropout']
    hid_dimension = model_params['hidden_dimension']
    multiplier = model_params['multiplier']

    if task == 'classification':
        loss_fn = SparseCategoricalCrossentropy()
        metrics = ['accuracy']
    elif task == 'regression':
        loss_fn = MeanAbsoluteError()
        metrics = ['mse']
        output_dimension = 1
    else:
        raise ValueError('Argument "task" must be one of "classification" ' \
                'or "regression"')

    if model_type == 'lstm' or model_type == 'gru':
        model = construct_rnn(input_dimension, output_dimension, model_type,
                n_cells, dropout, hid_dimension, model_name)
    elif model_type == 'lstm_cw' or model_type == 'gru_cw':
        model = construct_channel_wise_rnn(input_dimension, output_dimension,
                model_type, dropout, global_dropout, hid_dimension, multiplier,
                model_name)
    elif model_type == 'fcn':
        model = construct_fcn(input_dimension, output_dimension, hid_dimension,
                model_name)
    elif model_type == 'lstm_fcn':
        model = construct_lstm_fcn(input_dimension, output_dimension, dropout,
                hid_dimension, model_name)
    else:
        raise ValueError(f'Model type {model_type} is not supported.')

    if checkpoint_file:
        print(f"=> Loading weights from checkpoint: {checkpoint_file}")
        model.load_weights(os.path.join(checkpoints_dir, checkpoint_file))

    model.compile(optimizer=Adam(), loss=loss_fn, metrics=metrics)

    model.summary()

    return model


def squeeze_excite_block(X):
    """Squeeze and excitation block
    
    (Hu et al. 2019 - Squeeze-and-Excitation Networks)
    
    Args:
        X (tf.Tensor): Input tensor
        
    Returns:
        X (tf.Tensor): Transformed input tensor 
    """
    filters = X.shape[-1]
    r = 16

    # Squeeze
    se = GlobalAveragePooling1D()(X)
    se = Reshape((1, filters))(se)

    # Excite
    se = Dense(filters // r,  activation='relu',
            kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal',
            use_bias=False)(se)

    # Transform the input
    X = multiply([X, se])

    return X


class MetricsCallback(Callback):
    def __init__(self, model, task, training_data, validation_data,
            training_steps, validation_steps):
        """Callback to compute metrics after an epoch has ended
        
        Args:
            model (tf.keras.model): TensorFlow (Keras) model
            task (str): Classification or regression
            training_data (tf.data.Dataset)
            validation_data (tf.data.Dataset)
            training_steps (int)
            validation_steps (int)
        """
        self.model = model
        self.task = task
        self.training_data = training_data
        self.validation_data = validation_data

        self.training_steps = training_steps
        self.validation_steps = validation_steps

    def on_epoch_end(self, epoch, logs=None):
        """The callback

        Args:
            epoch (int): Identifier of the current epoch 
        """
        print('\n=> Predict on training data:\n')
        y_true, y_pred = [], []
        for batch, (x, y) in enumerate(self.training_data):
            if batch > self.training_steps:
                break

            if self.task == 'classification':
                y_pred.append(np.argmax(self.model.predict_on_batch(x), axis=1))
            else:
                y_pred.append(self.model.predict_on_batch(x))

            y_true.append(y.numpy())

        if self.task == 'classification':
            evaluate_classification_model(np.concatenate(y_true, axis=0),
                    np.concatenate(y_pred, axis=0))
        else:
            evaluate_regression_model(np.concatenate(y_true, axis=0),
                    np.concatenate(y_pred, axis=0))

        print('\n=> Predict on validation data:\n')
        y_true, y_pred = [], []
        for batch, (x, y) in enumerate(self.validation_data):
            if batch > self.validation_steps:
                break

            if self.task == 'classification':
                y_pred.append(np.argmax(self.model.predict_on_batch(x), axis=1))
            else:
                y_pred.append(self.model.predict_on_batch(x))

            y_true.append(y.numpy())

        if self.task == 'classification':
            evaluate_classification_model(np.concatenate(y_true, axis=0),
                    np.concatenate(y_pred, axis=0))
        else:
            evaluate_regression_model(np.concatenate(y_true, axis=0),
                    np.concatenate(y_pred, axis=0))


