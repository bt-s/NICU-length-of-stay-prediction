#!/usr/bin/python3

"""dnn.py

Script to run various types of Deep Neural Networks (DNNs) to predict the
remaining length-of-stay.
"""

__author__ = "Bas Straathof"

import tensorflow as tf

import argparse, csv, json, os
import numpy as np
import pandas as pd

from sys import argv
from datetime import datetime
from tqdm import tqdm
from sklearn.utils import resample

import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, \
        LearningRateScheduler, ModelCheckpoint, TensorBoard

from nicu_los.src.utils.modelling import construct_and_compile_model, \
        MetricsCallback
from nicu_los.src.utils.data_helpers import create_list_file, data_generator
from nicu_los.src.utils.evaluation import calculate_metric, \
        calculate_mean_absolute_error, calculate_confusion_matrix
from nicu_los.src.utils.visualization import plot_confusion_matrix 
        

def parse_cl_args():
    """Parses CL arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='data',
            help='Path to the data directories.')
    parser.add_argument( '--model-path', type=str,
            default='models/dnn',
            help='Path to the directory where the model should be saved.')
    parser.add_argument('--model-name', type=str, default='',
            help='The name of the model to be saved.')
    parser.add_argument('--model-type', type=str, default='lstm',
            help='The model to be loaded: "lstm", "lstm_cw", "gru", ' \
                    '"gru_cw", ΅fcn", or "fcn_lstm".')

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

    parser.add_argument('--no-mask-indicator', dest='mask_indicator',
            action='store_false')
    parser.add_argument('--no-gestational-age', dest='gestational_age',
            action='store_false')

    parser.add_argument('--training', dest='mode', action='store_const',
            const='training')
    parser.add_argument('--prediction', dest='mode', action='store_const',
            const='prediction')
    parser.add_argument('--evaluation', dest='mode', action='store_const',
            const='evaluation')

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

    parser.add_argument('--lr-scheduler',
            dest='lr_scheduler', action='store_true',
            help='Whether to use the learning rate scheduler.')

    parser.add_argument('--K', type=int, default=1000, help=('How often to ' +
        'perform bootstrap sampling without replacement when evaluating ' +
        'the model'))

    parser.add_argument('-c', '--config', type=str,
            default='nicu_los/config.json', help='Path to the config file')

    parser.add_argument('--debug-mode', dest='debug_mode',
            action='store_true')

    parser.set_defaults(enable_gpu=False, mode='training', coarse_targets=True,
            mask_indicator=True, metrics_callback=False, task='classification',
            allow_growth=False, gestational_age=True, lr_scheduler=False,
            debug_mode=False)

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
    mode = args.mode
    training_steps = args.training_steps
    validation_steps = args.validation_steps
    config = args.config
    debug_mode = args.debug_mode

    if debug_mode:
        print("==> DEBUG MODE")

    if args.enable_gpu:
        print('=> Using GPU(s)')
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        assert len(physical_devices) > 0, \
                "Not enough GPU hardware devices available"
        for i, _ in enumerate(physical_devices):
            config_ = tf.config.experimental.set_memory_growth(
                    physical_devices[i], args.allow_growth)
        print(f'=> Allow growth: {args.allow_growth}')
        strategy = tf.distribute.MirroredStrategy()

    else:
        print('=> Using CPU(s)')
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if mode == 'training':
        print(f'=> Training {model_name}') 
        print(f'=> Early stopping: {early_stopping}')
    elif mode == 'prediction':
        print(f'=> Predicting with {model_name}') 
    elif mode == 'evaluation':
        print(f'=> Evaluating {model_name}') 
    else:
        raise ValueError('Parameter "mode" must be one of: "training", ' +
                '"prediction", "evaluation"')
    print(f'=> Task: {task}')

    if task == "classification":
        print(f'=> Coarse targets: {coarse_targets}')
    print(f'=> Using mask: {mask}')
    print(f'=> Using gestational age variable: {gestational_age}')
    print(f'=> Batch size: {batch_size}')

    model_path = args.model_path
    if not os.path.exists(model_path):
            os.makedirs(model_path)

    if not debug_mode:
        checkpoints_dir = os.path.join(model_path, 'checkpoints')
    else:
        checkpoints_dir = os.path.join(model_path, 'debug' , 'checkpoints')
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    if mode == 'training':
        if not debug_mode:
            log_dir = os.path.join(model_path, 'logs', model_name + \
                    f'-batch{batch_size}-steps{training_steps}')
        else:
            log_dir = os.path.join(model_path, 'debug', 'logs', model_name + \
                    f'-batch{batch_size}-steps{training_steps}')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        if not debug_mode:
            log_dir_tb = os.path.join(log_dir,
                    datetime.now().strftime('%Y%m%d-%H%M%S'))
        else:
            log_dir_tb = os.path.join(log_dir, 'debug',
                    datetime.now().strftime('%Y%m%d-%H%M%S'))
        if not os.path.exists(log_dir_tb):
            os.makedirs(log_dir_tb)

    # Obtain the training variables
    with open(config) as f:
        config = json.load(f)
        variables = config['variables']
        bucket_boundaries = config['bucket_boundaries']
        bucket_sizes = [batch_size] * (len(bucket_boundaries)+1)

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

    if mode == 'training' or mode == 'prediction':
        if args.enable_gpu:
            with strategy.scope():
                model = construct_and_compile_model(model_type, model_name,
                        task, checkpoint_file, checkpoints_dir, model_params)
        else:
            model = construct_and_compile_model(model_type, model_name, task,
                    checkpoint_file, checkpoints_dir, model_params)

    if mode == 'training':
        train_list_file = os.path.join(data_path, 'train_list.txt')
        if not os.path.exists(train_list_file):
            with open(f'{data_path}/training_subjects.txt', 'r') as f:
                train_dirs = f.read().splitlines()
                create_list_file(train_dirs, train_list_file)

        val_list_file = os.path.join(data_path, 'val_list.txt')
        if not os.path.exists(val_list_file):
            with open(f'{data_path}/validation_subjects.txt', 'r') as f:
                val_dirs = f.read().splitlines()
                create_list_file(val_dirs, val_list_file)

        # Instantiate the training Dataset 
        train_data_generator = data_generator(train_list_file,
                coarse_targets=coarse_targets, mask=mask, 
                gestational_age=gestational_age, task=task)

        train_data = tf.data.Dataset.from_generator(lambda: 
                train_data_generator, output_types=(tf.float32, tf.int16),
                output_shapes=((None, len(variables)), []))

        train_data = train_data.apply(
            tf.data.experimental.bucket_by_sequence_length(
                element_length_func=lambda x, y: tf.shape(x)[0],
                bucket_batch_sizes=bucket_sizes,
                bucket_boundaries=bucket_boundaries))

        # Instantiate the validation Dataset 
        val_data_generator = data_generator(val_list_file,
                coarse_targets=coarse_targets, mask=mask, 
                gestational_age=gestational_age, task=task)

        val_data = tf.data.Dataset.from_generator(lambda: val_data_generator,
                output_types=(tf.float32, tf.int16),
                output_shapes=((None, len(variables)), []))

        val_data = val_data.apply(
            tf.data.experimental.bucket_by_sequence_length(
                element_length_func=lambda x, y: tf.shape(x)[0],
                bucket_batch_sizes=bucket_sizes,
                bucket_boundaries=bucket_boundaries))

        # Get callbacks
        checkpoint_path = os.path.join(checkpoints_dir, f'{model_name}-' + \
                f'batch{batch_size}-steps{training_steps}-epoch' + \
                '{epoch:02d}.h5')
        checkpoint_callback = ModelCheckpoint(checkpoint_path)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=log_dir_tb, histogram_freq=1)
        logger_callback = CSVLogger(os.path.join(log_dir, 'logs.csv'),
                append=True)

        callbacks = [checkpoint_callback, logger_callback,
                tensorboard_callback]

        if args.metrics_callback:
            metrics_callback = MetricsCallback(model, task, train_data,
                    val_data, training_steps, validation_steps)
            callbacks.append(metrics_callback)

        if early_stopping:
            early_stopping_callback = EarlyStopping(monitor='val_loss',
                    min_delta=0, patience=early_stopping)
            callbacks.append(early_stopping_callback)

        if args.lr_scheduler:
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
            validation_steps=validation_steps, callbacks=callbacks, workers=20,
            use_multiprocessing=True, max_queue_size=20)

    elif mode == 'prediction': 
        if not debug_mode:
            predictions_dir = os.path.join(model_path, 'predictions',
                    checkpoint_file)
        else:
            predictions_dir = os.path.join(model_path, 'debug', 'predictions',
                    checkpoint_file)
        if not os.path.exists(predictions_dir):
            os.makedirs(predictions_dir)
        f_name_predictions = os.path.join(predictions_dir, f'predictions.csv')

        test_list_file = os.path.join(data_path, 'test_list.txt')
        if not os.path.exists(test_list_file):
            with open(f'{data_path}/test_subjects.txt', 'r') as f:
                test_dirs = f.read().splitlines()
                create_list_file(test_dirs, test_list_file)

        with open(test_list_file, 'r') as f:
            n_test_seqs = len(f.read().splitlines())
            total_n_steps = n_test_seqs//batch_size

        # Instantiate the test Dataset 
        test_data_generator = data_generator(test_list_file,
                coarse_targets=coarse_targets, mask=mask, 
                gestational_age=gestational_age, task=task, shuffle=True)

        test_data = tf.data.Dataset.from_generator(lambda: 
                test_data_generator, output_types=(tf.float32, tf.int16),
                output_shapes=((None, len(variables)), []))

        test_data = test_data.apply(
            tf.data.experimental.bucket_by_sequence_length(
                element_length_func=lambda x, y: tf.shape(x)[0],
                bucket_batch_sizes=bucket_sizes,
                bucket_boundaries=bucket_boundaries))

        y_true, y_pred = [], []
        for batch, (x, y) in enumerate(tqdm(test_data, total=(total_n_steps))):
            if batch == total_n_steps:
                break

            y_true.append(y.numpy())
            if task == "regression":
                y_pred.append(model.predict_on_batch(x))
            else: # classification
                y_pred.append(np.argmax(model.predict_on_batch(x), axis=1))

        y_pred = np.concatenate(y_pred).ravel()
        y_pred = np.maximum(y_pred, 0) # remaining LOS can't be negative
        y_true = np.concatenate(y_true).ravel()

        print(f'=> Writing results to {f_name_predictions}')
        with open(f_name_predictions, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['True labels', 'Predictions'])
            writer.writerows(zip(y_true, y_pred))

    elif mode == 'evaluation': 
        if not debug_mode:
            f_name_predictions = os.path.join(model_path, 'predictions',
                    checkpoint_file, 'predictions.csv')
        else:
            f_name_predictions = os.path.join(model_path, 'debug',
                    'predictions', checkpoint_file, 'predictions.csv')
        if not os.path.exists(f_name_predictions):
            raise FileNotFoundError(f"File note found: could not find " +
                    f"{f_name_predictions} make sure to predict first.")

        if not debug_mode:
            results_dir = os.path.join(model_path, 'results',
                    checkpoint_file)
        else:
            results_dir = os.path.join(model_path, 'debug', 'results',
                    checkpoint_file)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        f_name_results = os.path.join(results_dir, f'results.json')

        if task == 'classification':
            f_name_confusion_matrix = os.path.join(results_dir, f'cm.pdf')
            f_name_confusion_matrix_normalized = os.path.join(results_dir,
                    f'cm_normalized.pdf')

        # Open the dataframe containing all predicitons on the test set
        df_pred = pd.read_csv(f_name_predictions, index_col=False)

        print(f'=> K={args.K} bootstrapping rounds')

        if task == 'regression':
            metrics = ['MAE']
        else:
            metrics = ['accuracy', 'kappa', 'recall', 'precision', 'f1']

        results = {'iters': args.K}
        
        for m in metrics:
            results[m] = dict()
            results[m]['iters'] = []

        for k in tqdm(range(args.K)):
            y_true = df_pred['True labels'].to_numpy()
            y_true = resample(y_true, random_state=k)
            y_pred = df_pred['Predictions'].to_numpy()
            y_pred = resample(y_pred, random_state=k)

            for m in metrics:
                results[m]['iters'].append(calculate_metric(y_true, y_pred,
                    metric=m, verbose=False))

        for m in metrics:
            iters = results[m]['iters']
            results[m]['mean'] = np.mean(iters)
            results[m]['median'] = np.median(iters)
            results[m]['std'] = np.std(iters)
            results[m]['2.5 percentile'] = np.percentile(iters, 2.5)
            results[m]['97.5 percentile'] = np.percentile(iters, 97.5)
            del results[m]['iters']

        if task == 'classification':
            # Create and plot confusion matrix
            y_true = df_pred['True labels'].to_numpy()
            y_pred = df_pred['Predictions'].to_numpy()

            cm = calculate_confusion_matrix(y_true, y_pred)
            cm_normalized = calculate_confusion_matrix(y_true, y_pred,
                    normalize='pred')
        
            plot_confusion_matrix(cm, output_dimension,
                    f_name_confusion_matrix)
            plot_confusion_matrix(cm_normalized, output_dimension, 
                    f_name_confusion_matrix_normalized)

            results['confusion matrix'] = cm.tolist()
            results['confusion matrix normalized'] = cm_normalized.tolist()

        print(f'=> Writing results to {f_name_results}')
        with open(f_name_results, 'w') as f:
            json.dump(results, f)


if __name__ == '__main__':
    main(parse_cl_args())

