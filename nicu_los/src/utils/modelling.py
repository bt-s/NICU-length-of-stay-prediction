#!/usr/bin/python3

"""modelling.py

Various utility functions for modelling
"""

__author__ = "Bas Straathof"

import os 

import numpy as np

import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, BatchNormalization, \
        Bidirectional, concatenate, Conv1D, Dense, Dropout, \
        GlobalAveragePooling1D, GRU, Input, LSTM, Masking, \
        SpatialDropout1D
from tensorflow.keras.losses import MeanAbsoluteError, SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

from nicu_los.src.utils.evaluation import evaluate_classification_model, \
        evaluate_regression_model
from nicu_los.src.utils.custom_keras_layers import ApplyMask, squeeze_excite_block, \
       Slice 


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
    mask = Masking().compute_mask(inputs)

    X = Conv1D(128, 8, padding='same',
            kernel_initializer='he_uniform')(inputs)
    X = Activation('relu')(X)
    X = BatchNormalization()(X)
    X = SpatialDropout1D(0.5)(X)
    X = ApplyMask()(X, mask)
    X = squeeze_excite_block(X, mask)

    X = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(X)
    X = Activation('relu')(X)
    X = BatchNormalization()(X)
    X = SpatialDropout1D(0.5)(X)
    X = ApplyMask()(X, mask)
    X = squeeze_excite_block(X, mask)

    X = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(X)
    X = Activation('relu')(X)
    X = BatchNormalization()(X)

    X = GlobalAveragePooling1D()(X, mask)

    if output_dimension != 1:
        # Classification
        outputs = Dense(units=output_dimension, activation='softmax')(X)
    else:
        # Regression 
        outputs = Dense(units=output_dimension)(X)

    model = Model(inputs=inputs, outputs=outputs, name=model_name)

    return model


def construct_fcn_originial(input_dimension, output_dimension,
        hid_dimension_lstm=8, model_name=""):
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


def construct_lstm_fcn_original(input_dimension, output_dimension, dropout=0.8,
        hid_dimension_lstm=8, model_name=""):
    """Construct an LSTM-FCN model 
    
    Architecture as described in:
        Karim et al. 2019 - Multivariate LSTM-FCNs for time series classification

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


def construct_lstm_fcn(input_dimension, output_dimension, dropout=0.8,
        hid_dimension_lstm=8, model_name=""):
    """Construct a (modified) LSTM-FCN model 
    
    Modified architecture:
       - Perform batch normalization after ReLU activation
       - Use SpatialDropout1D in the convolutional blocks to reduce overfitting

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
    X2 = Activation('relu')(X2)
    X2 = BatchNormalization()(X2)
    X2 = SpatialDropout1D(0.5)(X2)
    X2 = squeeze_excite_block(X2)

    X2 = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(X2)
    X2 = Activation('relu')(X2)
    X2 = BatchNormalization()(X2)
    X2 = SpatialDropout1D(0.5)(X2)
    X2 = squeeze_excite_block(X2)

    X2 = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(X2)
    X2 = Activation('relu')(X2)
    X2 = BatchNormalization()(X2)

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

