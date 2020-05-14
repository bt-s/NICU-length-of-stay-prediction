#!/usr/bin/python3

"""rnn.py

Script to run various types of Recurrent Neural Networks (RNNs) to predict the
remaining length-of-stay.
"""

__author__ = "Bas Straathof"

import tensorflow as tf

import argparse, json, os
import numpy as np

from sys import argv
from datetime import datetime
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, \
        ModelCheckpoint, TensorBoard

from nicu_los.src.utils.modelling import construct_and_compile_model, \
        create_list_file, data_generator, MetricsCallback
from nicu_los.src.utils.evaluation import evaluate_classification_model
        
from nicu_los.src.utils.readers import TimeSeriesReader


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
    parser.add_argument('--epochs', type=int, default=20,
            help='Number of training epochs.')
    parser.add_argument('--training-steps', type=int, default=2500,
            help='Training steps per training epoch.')
    parser.add_argument('--validation-steps', type=int, default=1000,
            help='Validation steps per training epoch.')
    parser.add_argument('--early-stopping', type=int, default=0,
            help=('Whether to use the early stopping callback. This number ' \
                    'indicates the patience, i.e. the number of epochs that ' \
                    'the validation loss is allowed to be lower than the '
                    'previous one.'))

    parser.add_argument('--dropout', type=float, default=0.0,
            help='The amount of dropout to be used.')
    parser.add_argument('--hidden-dimension', type=int, default=64,
            help='The hidden dimension per layer of the RNN.')
    parser.add_argument('--n-cells', type=int, default=1,
            help='The number of cells in the RNN.')

    parser.add_argument('--mask-indicatior', dest='mask_indicator',
            action='store_true')
    parser.add_argument('--no-mask-indicator', dest='mask_indicator',
            action='store_false')

    parser.add_argument('--training', dest='training', action='store_true')
    parser.add_argument('--testing', dest='training', action='store_false')

    parser.add_argument('--metrics-callback', dest='metrics_callback', action='store_true')
    parser.add_argument('--no-metrics-callback', dest='metrics_callback', action='store_false')

    parser.add_argument('--coarse-targets', dest='coarse_targets',
            action='store_true')
    parser.add_argument('--no-coarse-targets', dest='coarse_targets',
            action='store_false')

    parser.add_argument('--enable-gpu', dest='enable_gpu',
            action='store_true')
    parser.add_argument('--disable-gpu', dest='enable_gpu',
            action='store_false')

    parser.set_defaults(enable_gpu=False, training=True, coarse_targets=True,
            mask_indicator=True, metrics_callback=False)

    return parser.parse_args(argv[1:])


def main(args):
    if args.enable_gpu:
        print('=> Using GPU(s)')
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
        for i, _ in enumerate(physical_devices):
            config = tf.config.experimental.set_memory_growth(physical_devices[i], True)
        strategy = tf.distribute.MirroredStrategy()

    else:
        print('=> Using CPU(s)')
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    coarse_targets = args.coarse_targets
    model_name = args.model_name
    model_type = args.model_type

    early_stopping = args.early_stopping
    training = args.training

    if training:
        print(f'=> Training {model_name}') 
        print(f'=> Early stopping: {early_stopping}')
    else:
        print(f'=> Evaluating {model_name}') 

    data_path = args.data_path
    model_path = args.model_path
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    checkpoint_file = args.checkpoint_file
    checkpoints_dir = os.path.join(model_path, 'checkpoints')
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    if training:
        log_dir = os.path.join(model_path, 'logs', model_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_dir_tb = os.path.join(log_dir, datetime.now().strftime('%Y%m%d-%H%M%S'))
        if not os.path.exists(log_dir_tb):
            os.makedirs(log_dir_tb)

    mask = args.mask_indicator
    training_steps = args.training_steps
    validation_steps = args.validation_steps
    initial_epoch = args.initial_epoch
    batch_size = args.batch_size

    print(f'=> Coarse targets: {coarse_targets}')
    print(f'=> Using mask: {mask}')
    print(f'=> Batch size: {batch_size}')

    checkpoint_path = os.path.join(checkpoints_dir, f'{model_name}-' + \
            f'batch{batch_size}-steps{training_steps}-epoch' + \
            '{epoch:02d}.h5')

    # Obtain the training variables
    with open('nicu_los/config.json') as f:
        config = json.load(f)
        variables = config['variables']

        if mask:
            variables = variables + ['mask_' + v for v in variables]

    if coarse_targets:
        output_dimension = 3
    else:
        output_dimension = 10

    model_params = {
        'input_dimension': len(variables),
        'output_dimension': output_dimension,
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
                log_dir=log_dir_tb, histogram_freq=1)
        logger_callback = CSVLogger(os.path.join(model_path, 'logs', model_name,
            'logs.csv'))
        callbacks = [checkpoint_callback, logger_callback, tensorboard_callback]

        if args.metrics_callback:
            metrics_callback = MetricsCallback(model, train_data,
                    val_data, training_steps, validation_steps)
            callbacks.append(metrics_callback)

        if early_stopping:
            early_stopping_callback = EarlyStopping(monitor='val_loss',
                    min_delta=0, patience=args.early_stopping)
            callbacks.append(early_stopping_callback)

        print(f'=> Fitting the model')
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
        batch_size = 8 
        steps = 1000
        shuffle = False

        if not os.path.exists(test_list_file):
            with open(f'{data_path}/test_subjects.txt', 'r') as f:
                test_dirs = f.read().splitlines()
                create_list_file(test_dirs, test_list_file)

        test_data = tf.data.Dataset.from_generator(data_generator,
                args=[test_list_file, steps, batch_size, "classification",
                    coarse_targets, mask, shuffle],
                output_types=(tf.float32, tf.int16),
                output_shapes=((batch_size, None, len(variables)),
                    (batch_size,)))

        y_true, y_pred = [], []

        # Get the number of sequences by instantiating the test reader
        test_reader = TimeSeriesReader(test_list_file, coarse_targets=coarse_targets,
                mask=mask)
        n_sequences = test_reader.get_number_of_sequences()
        
        for batch, (x, y) in enumerate(tqdm(test_data, total=n_sequences/batch_size)):
            if batch == 1000:#> n_sequences/batch_size:
                break

            y_pred.append(np.argmax(model.predict_on_batch(x), axis=1))
            y_true.append(y.numpy())

        y_pred = np.concatenate(y_pred).ravel()
        y_true = np.concatenate(y_true).ravel()

        evaluate_classification_model(y_true, y_pred)


if __name__ == '__main__':
    main(parse_cl_args())

