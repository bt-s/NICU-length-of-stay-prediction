#!/usr/bin/python3

"""rnn.py

Script to run various types of Recurrent Neural Networks (RNNs) to predict the
remaining length-of-stay.
"""

__author__ = "Bas Straathof"

import tensorflow as tf

import argparse, json, os
from sys import argv
from datetime import datetime
import numpy as np

import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, \
        ModelCheckpoint, TensorBoard

from nicu_los.src.utils.modelling_utils import construct_and_compile_model, \
        create_list_file, data_generator, evaluate_classification_model, \
        MetricsCallback


def parse_cl_args():
    """Parses CL arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='data',
            help='Path to the data directories.')
    parser.add_argument( '--model-path', type=str,
            default='models/rnn',
            help='Path to the directory where the model should be saved.')
    parser.add_argument('--model-name', type=str, default='',
            help='The name of the model to be saved.')
    parser.add_argument('--model-type', type=str, default='lstm',
            help='The model to be loaded. Either "lstm" or "gru".')

    parser.add_argument('--checkpoint-file', type=str, default="",
            help='File from which to load the model weights.')
    parser.add_argument('--initial-epoch', type=int, default=0,
            help='The starting epoch if loading a checkpoint file.')

    parser.add_argument('--batch-size', type=int, default=8,
            help='Training batch size.')
    parser.add_argument('--coarse-targets', type=int, default=0,
            help='Whether to use coarse target labels.')
    parser.add_argument('--epochs', type=int, default=100,
            help='Number of training epochs.')
    parser.add_argument('--training-steps', type=int, default=20,
            help='Training steps per training epoch.')
    parser.add_argument('--validation-steps', type=int, default=10,
            help='Validation steps per training epoch.')
    parser.add_argument('--early-stopping', type=int, default=0,
            help=('Whether to use the early stopping callback. This number ' \
                    'indicates the patience, i.e. the number of epochs that ' \
                    'the validation loss is allowed to be lower than the '
                    'previous one.'))
    parser.add_argument('--mask-indicator', type=int, default=1,
            help='Whether to use missinggness indicator mask variables.')

    parser.add_argument('--dropout', type=float, default=0.0,
            help='The amount of dropout to be used.')
    parser.add_argument('--hidden-dimension', type=int, default=64,
            help='The hidden dimension per layer of the RNN.')
    parser.add_argument('--n-cells', type=int, default=1,
            help='The number of cells in the RNN.')

    parser.add_argument('--training', type=int, default=1,
            help='Whether the current phase is the training phase.')
    parser.add_argument('--enable-gpu', type=int, default=0,
            help='Whether the GPU(s) should be enabled.')

    return parser.parse_args(argv[1:])


def main(args):
    if args.enable_gpu:
        strategy = tf.distribute.MirroredStrategy()
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    coarse_targets = args.coarse_targets
    training = args.training
    model_name = args.model_name
    model_type = args.model_type
    if training:
        print(f'Training {model_name}')

    data_path = args.data_path
    model_path = args.model_path
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    checkpoint_file = args.checkpoint_file
    checkpoints_dir = os.path.join(model_path, 'checkpoints')
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    log_dir = model_path + '/logs/' + datetime.now().strftime('%Y%m%d-%H%M%S')

    loss_log_dir = os.path.join(model_path, 'logs', model_name)
    if not os.path.exists(loss_log_dir):
        os.makedirs(loss_log_dir)

    mask = args.mask_indicator
    batch_size = args.batch_size
    training_steps = args.training_steps
    validation_steps = args.validation_steps
    initial_epoch = args.initial_epoch

    checkpoint_path = os.path.join(checkpoints_dir, f'{model_name}-' + \
            f'batch{batch_size}-steps{training_steps}-epoch' + \
            '{epoch:02d}.h5')

    # Obtain the training variables
    with open('nicu_los/config.json') as f:
        config = json.load(f)
        variables = config['variables']

        if mask:
            variables = variables + ['mask_' + v for v in variables]

    model_params = {
        'input_dimension': len(variables),
        'hidden_dimension': args.hidden_dimension,
        'dropout': args.dropout,
        'n_cells': args.n_cells}

    if args.enable_gpu:
        with strategy.scope():
            model = construct_and_compile_model(model_type, model_name,
                    checkpoint_file, checkpoints_dir, model_params)
    else:
        model = construct_and_compile_model(model_type, model_name,
                checkpoint_file, checkpoints_dir, model_params)

    model.summary()

    if training:
        train_list_file = os.path.join(data_path, 'train_list.txt')
        val_list_file = os.path.join(data_path, 'val_list.txt')

        if not os.path.exists(train_list_file):
            with open(f'{data_path}/training_subjects.txt', 'r') as f:
                train_dirs = f.read().splitlines()
                create_list_file(train_dirs, train_list_file)

        if not os.path.exists(val_list_file):
            with open(f'{data_path}/validation_subjects.txt', 'r') as f:
                val_dirs = f.read().splitlines()
                create_list_file(val_dirs, val_list_file)

        train_data = tf.data.Dataset.from_generator(data_generator,
                args=[train_list_file, training_steps, batch_size,
                    "classification", coarse_targets, mask],
                output_types=(tf.float32, tf.int16),
                output_shapes=((batch_size, None, len(variables)),
                    (batch_size,)))

        val_data = tf.data.Dataset.from_generator(data_generator,
                args=[val_list_file, validation_steps, batch_size,
                    "classification", coarse_targets, mask],
                output_types=(tf.float32, tf.int16),
                output_shapes=((batch_size, None, len(variables)),
                    (batch_size,)))

        # Get callbacks
        checkpoint_callback = ModelCheckpoint(checkpoint_path)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=log_dir, histogram_freq=1)
        logger_callback = CSVLogger(os.path.join(model_path, 'logs', model_name,
            'logs.csv'))
        metrics_callback = MetricsCallback(model, train_data,
                val_data, training_steps, validation_steps)
        callbacks = [checkpoint_callback, logger_callback, metrics_callback,
                tensorboard_callback]

        if args.early_stopping:
            early_stopping_callback = EarlyStopping(monitor='val_loss',
                    min_delta=0, patience=args.early_stopping)
            callbacks.append(early_stopping_callback)

        # Fit the model
        model.fit(
            train_data,
            validation_data=val_data,
            epochs=args.epochs,
            initial_epoch=initial_epoch,
            steps_per_epoch=training_steps,
            validation_steps=validation_steps,
            callbacks=callbacks)

    else:
        test_list_file = os.path.join(data_path, 'test_list.txt')

        if not os.path.exists(test_list_file):
            with open(f'{data_path}/test_subjects.txt', 'r') as f:
                test_dirs = f.read().splitlines()
                create_list_file(test_dirs, test_list_file)

        test_data = tf.data.Dataset.from_generator(data_generator,
                args=[test_list_file, 0, batch_size, "classification",
                    coarse_targets, mask],
                output_types=(tf.float32, tf.int16),
                output_shapes=((None, len(variables)), ()))

        y_true = []
        y_pred = []
        for batch, (x, y) in enumerate(test_data):
            y_pred.append(np.argmax(model.predict_on_batch(x), axis=1))
            y_true.append(y.numpy())


        y_pred = np.concatenate(y_pred).ravel()
        y_true = np.concatenate(y_true).ravel()

        evaluate_classification_model(y_true, y_pred)


if __name__ == '__main__':
    main(parse_cl_args())

