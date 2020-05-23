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
        LearningRateScheduler, ModelCheckpoint, TensorBoard

from nicu_los.src.utils.modelling import construct_and_compile_model, \
        create_list_file, data_generator, MetricsCallback
from nicu_los.src.utils.evaluation import evaluate_classification_model, \
        evaluate_regression_model
        
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
    parser.add_argument('--training-steps', type=int, default=None,
            help='Training steps per training epoch.')
    parser.add_argument('--validation-steps', type=int, default=None,
            help='Validation steps per training epoch.')
    parser.add_argument('--early-stopping', type=int, default=0,
            help=('Whether to use the early stopping callback. This number ' \
                    'indicates the patience, i.e. the number of epochs that ' \
                    'the validation loss is allowed to be lower than the '
                    'previous one.'))

    parser.add_argument('--dropout', type=float, default=0.0,
            help='The amount of dropout to be used.')
    parser.add_argument('--global-dropout', type=float, default=0.0,
            help='The amount of dropout to be used in the global dropout " \
                    "layer of a multi-channel RNN.')
    parser.add_argument('--hidden-dimension', type=int, default=64,
            help='The hidden dimension per layer of the RNN.')
    parser.add_argument('--multiplier', type=int, default=4,
            help='Multiplier of the hidden dimension of the global cell in '\
                    'channel-wise RNN.')
    parser.add_argument('--n-cells', type=int, default=1,
            help='The number of cells in the RNN " \
                    " (does not apply to channel-wise RNN).')

    parser.add_argument('--mask-indicatior', dest='mask_indicator',
            action='store_true')
    parser.add_argument('--no-mask-indicator', dest='mask_indicator',
            action='store_false')
    parser.add_argument('--no-gestational-age', dest='gestational_age',
            action='store_false')

    parser.add_argument('--training', dest='training', action='store_true')
    parser.add_argument('--testing', dest='training', action='store_false')

    parser.add_argument('--metrics-callback', dest='metrics_callback',
            action='store_true')

    parser.add_argument('--coarse-targets', dest='coarse_targets',
            action='store_true')
    parser.add_argument('--fine-targets', dest='coarse_targets',
            action='store_false')

    parser.add_argument('--enable-gpu', dest='enable_gpu',
            action='store_true')

    parser.add_argument('--allow-growth', dest='allow_growth',
            action='store_true', help=('Whether to allow growing memory for ' +
                'the GPU'))

    parser.add_argument('--regression', dest='task', action='store_const',
            const='regression')

    parser.set_defaults(enable_gpu=False, training=True, coarse_targets=True,
            mask_indicator=True, metrics_callback=False, task='classification',
            allow_growth=False, gestational_age=True)

    return parser.parse_args(argv[1:])


def main(args):
    batch_size = args.batch_size
    checkpoint_file = args.checkpoint_file
    coarse_targets = args.coarse_targets
    data_path = args.data_path
    early_stopping = args.early_stopping
    gestational_age = args.gestational_age
    mask = args.mask_indicator
    model_name = args.model_name
    model_type = args.model_type
    task = args.task
    training = args.training
    training_steps = args.training_steps
    validation_steps = args.validation_steps

    if args.enable_gpu:
        print('=> Using GPU(s)')
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        assert len(physical_devices) > 0, \
                "Not enough GPU hardware devices available"
        for i, _ in enumerate(physical_devices):
            config = tf.config.experimental.set_memory_growth(
                    physical_devices[i], args.allow_growth)
        print(f'=> Allow growth: {args.allow_growth}')
        strategy = tf.distribute.MirroredStrategy()

    else:
        print('=> Using CPU(s)')
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if training:
        print(f'=> Training {model_name}') 
        print(f'=> Early stopping: {early_stopping}')
        print(f'=> Task: {task}')
    else:
        print(f'=> Evaluating {model_name}') 

    if task == "classification":
        print(f'=> Coarse targets: {coarse_targets}')
    print(f'=> Using mask: {mask}')
    print(f'=> Using gestational age variable: {gestational_age}')
    print(f'=> Batch size: {batch_size}')

    model_path = args.model_path
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    checkpoints_dir = os.path.join(model_path, 'checkpoints')
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    if training:
        log_dir = os.path.join(model_path, 'logs', model_name + \
                f'-batch{batch_size}-steps{training_steps}')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_dir_tb = os.path.join(log_dir,
                datetime.now().strftime('%Y%m%d-%H%M%S'))
        if not os.path.exists(log_dir_tb):
            os.makedirs(log_dir_tb)

    # Obtain the training variables
    with open('nicu_los/config.json') as f:
        config = json.load(f)
        variables = config['variables']

        if not gestational_age and "GESTATIONAL_AGE_DAYS" in variables:
            variables.remove("GESTATIONAL_AGE_DAYS")

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
        'global_dropout': args.global_dropout,
        'multiplier': args.multiplier,
        'n_cells': args.n_cells}

    if args.enable_gpu:
        with strategy.scope():
            model = construct_and_compile_model(model_type, model_name, task,
                    checkpoint_file, checkpoints_dir, model_params)
    else:
        model = construct_and_compile_model(model_type, model_name, task,
                checkpoint_file, checkpoints_dir, model_params)

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

        # Instantiate the training and validation readers
        train_reader = TimeSeriesReader(train_list_file,
                coarse_targets=coarse_targets, mask=mask, name="Train reader")
        val_reader = TimeSeriesReader(val_list_file,
                coarse_targets=coarse_targets, mask=mask,
                name="Validation reader")

        train_data_generator = data_generator(train_reader, training_steps,
                batch_size, task)
        val_data_generator = data_generator(val_reader, validation_steps,
                batch_size, task)

        train_data = tf.data.Dataset.from_generator(lambda: train_data_generator,
                output_types=(tf.float32, tf.int16),
                output_shapes=((batch_size, None, len(variables)),
                    (batch_size,)))

        val_data = tf.data.Dataset.from_generator(lambda: val_data_generator,
                output_types=(tf.float32, tf.int16),
                output_shapes=((batch_size, None, len(variables)),
                    (batch_size,)))

        # Get callbacks
        checkpoint_path = os.path.join(checkpoints_dir, f'{model_name}-' + \
                f'batch{batch_size}-steps{training_steps}-epoch' + \
                '{epoch:02d}.h5')
        checkpoint_callback = ModelCheckpoint(checkpoint_path)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=log_dir_tb, histogram_freq=1)
        logger_callback = CSVLogger(os.path.join(log_dir, 'logs.csv'))

        callbacks = [checkpoint_callback, logger_callback, tensorboard_callback]

        if args.metrics_callback:
            metrics_callback = MetricsCallback(model, task, train_data,
                    val_data, training_steps, validation_steps)
            callbacks.append(metrics_callback)

        if early_stopping:
            early_stopping_callback = EarlyStopping(monitor='val_loss',
                    min_delta=0, patience=early_stopping)
            callbacks.append(early_stopping_callback)

        if task == "regression":
            def lr_schedule(epoch):
                if epoch < 3:
                    lr = 0.01
                elif epoch < 6:
                    lr = 0.005
                else:
                    lr = 0.001
                return lr
            lr_scheduler = LearningRateScheduler(lr_schedule)
            print("=> Create a learning rate scheduler")
            callbacks.append(lr_scheduler)

        print(f'=> Fitting the model')
        model.fit(train_data, validation_data=val_data, epochs=args.epochs,
            initial_epoch=args.initial_epoch, steps_per_epoch=training_steps,
            validation_steps=validation_steps, callbacks=callbacks)

    else:
        test_list_file = os.path.join(data_path, 'test_list.txt')
        batch_size = 16 
        test_steps = 1000
        shuffle = False

        if not os.path.exists(test_list_file):
            with open(f'{data_path}/test_subjects.txt', 'r') as f:
                test_dirs = f.read().splitlines()
                create_list_file(test_dirs, test_list_file)

        test_reader = TimeSeriesReader(test_list_file,
                coarse_targets=coarse_targets, mask=mask, name="Test reader")
        test_data_generator = data_generator(test_reader, test_steps,
                batch_size, task)

        test_data = tf.data.Dataset.from_generator(test_data_generator,
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

        if task == 'classification':
            evaluate_classification_model(y_true, y_pred)
        else:
            evaluate_regression_model(y_true, y_pred)


if __name__ == '__main__':
    main(parse_cl_args())

