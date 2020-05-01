#!/usr/bin/python3

"""simple_gru.py

Implementation of a simple LSTM model to predict the remaining length-of-stay.
"""

__author__ = "Bas Straathof"

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, \
        ModelCheckpoint, TensorBoard
from tensorflow.keras.losses import SparseCategoricalCrossentropy

import argparse, json, os
from sys import argv
from datetime import datetime
import numpy as np

from ..utils.modelling_utils import create_list_file, data_generator, \
        get_bucket_by_seq_len, construct_simple_gru
from ..utils.evaluation_utils import evaluate_classification_model


def parse_cl_args():
    """Parses CL arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='data',
            help='Path to the data directories.')
    parser.add_argument( '--models-path', type=str,
            default='models/simple_gru',
            help='Path to the simple LSTM models directory.')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size.')
    parser.add_argument('--mask-indicator', type=int, default=1,
            help='Whether to use missinggness indicator mask variables.')
    parser.add_argument('--training-steps', type=int, default=2000,
            help='Training steps per epoch.')
    parser.add_argument('--validation-steps', type=int, default=1000,
            help='Validation steps per epoch.')
    parser.add_argument('--epochs', type=int, default=100,
            help='Epochs.')
    parser.add_argument('--training', type=int, default=0,
            help='Whether the current phase is the training phase.')
    parser.add_argument('--checkpoint-file', type=str, default="",
            help='File from which to load the model weights.')
    parser.add_argument('--initial-epoch', type=int, default=0,
            help='The starting epoch if loading a checkpoint file.')
    parser.add_argument('--model-name', type=str, default='',
            help='The name of the model to be trained.')
    parser.add_argument('--early-stopping', type=int, default=0,
            help='Whether to use the early stopping callback.')

    return parser.parse_args(argv[1:])


def main(args):
    strategy = tf.distribute.MirroredStrategy()
    training = args.training
    model_name = args.model_name
    if training:
        print(f'Training {model_name}')

    data_path = args.data_path
    models_path = args.models_path
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    checkpoints_dir = os.path.join(models_path, 'checkpoints')
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    log_dir = models_path + '/logs/' + datetime.now().strftime('%Y%m%d-%H%M%S')

    mask = args.mask_indicator
    batch_size = args.batch_size
    training_steps = args.training_steps
    validation_steps = args.validation_steps
    initial_epoch = args.initial_epoch
    training = args.training

    checkpoint_path = os.path.join(checkpoints_dir, f'{model_name}-' + \
            f'batch{batch_size}-steps{training_steps}-epoch' + \
            '{epoch:02d}.h5')

    bucket_by_seq_len = get_bucket_by_seq_len(batch_size)

    # Obtain the training variables
    with open('nicu_los/config.json') as f:
        config = json.load(f)
        variables = config['variables']

        if mask:
            variables = variables + ['mask_' + v for v in variables]

    with strategy.scope():
        # Construct the model
        model = construct_simple_gru()
        if args.checkpoint_file:
            model.load_weights(os.path.join(checkpoints_dir,
                args.checkpoint_file))

        # Compile the model
        model.compile(
                optimizer=Adam(),
                loss=SparseCategoricalCrossentropy(),
                metrics=['accuracy'])

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
                args=[train_list_file, training_steps, batch_size, mask],
                output_types=(tf.float32, tf.int16),
                output_shapes=((None, len(variables)), ()))

        val_data = tf.data.Dataset.from_generator(data_generator,
                args=[val_list_file, validation_steps, batch_size, mask],
                output_types=(tf.float32, tf.int16),
                output_shapes=((None, len(variables)), ()))

        # Sort the data set batches by sequence length
        train_data = train_data.apply(bucket_by_seq_len)
        val_data = val_data.apply(bucket_by_seq_len)

        # Callbacks
        checkpoint_callback = ModelCheckpoint(checkpoint_path)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                histogram_freq=1)
        logger_callback = CSVLogger(os.path.join(models_path, 'logs',
            'logs.csv'))
        callbacks = [checkpoint_callback, logger_callback, tensorboard_callback]

        if args.early_stopping:
            early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0,
                    patience=0)
            callbacks.append(early_stopping_callback)

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
                args=[test_list_file, 0, batch_size, mask],
                output_types=(tf.float32, tf.int16),
                output_shapes=((None, len(variables)), ()))

        test_data = test_data.apply(bucket_by_seq_len)
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

