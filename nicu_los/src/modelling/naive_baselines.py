#!/usr/bin/python3

"""naive_baselines.py

Script to test two naive baselines:
    (1) Always predict the mean
    (2) Always predict the median
"""

__author__ = "Bas Straathof"

import argparse, os, pickle
from sklearn.utils import resample
from tqdm import tqdm

import numpy as np
import pandas as pd

from sys import argv
from datetime import datetime

from nicu_los.src.utils.data_helpers import get_baseline_datasets
from nicu_los.src.utils.evaluation import calculate_metric, \
        evaluate_classification_model, evaluate_regression_model


def parse_cl_args():
    """Parses CL arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-sp', '--subjects-path', type=str,
            default='data', help='Path to the subjects directories.')
    parser.add_argument('-mp', '--models-path', type=str,
            default='models/naive_baselines/',
            help='Path to the models directory.')
    parser.add_argument('-mn', '--model-name', type=str, default="",
            help='Name of the model.')
    parser.add_argument('--model-task', type=str, default='classification',
            help='Task; either "classification" or "regression".')

    parser.add_argument('--coarse-targets', dest='coarse_targets',
            action='store_true')
    parser.add_argument('--fine-targets', dest='coarse_targets',
            action='store_false')

    parser.add_argument('--friedman', dest='friedman', action='store_true')

    parser.add_argument('-c', '--config', type=str,
            default='nicu_los/config.json', help='Path to the config file')

    parser.set_defaults(coarse_targets=True, friedman=False)

    return parser.parse_args(argv[1:])


def main(args):
    if not os.path.exists(args.models_path):
        os.makedirs(args.models_path)

    data_path = args.subjects_path
    friedman = args.friedman
    model_name = args.model_name
    model_task = args.model_task
    coarse_targets = args.coarse_targets
    config = args.config

    with open(f'{data_path}/training_subjects.txt', 'r') as f:
        train_dirs = f.read().splitlines()
    with open(f'{data_path}/validation_subjects.txt', 'r') as f:
        val_dirs = f.read().splitlines()
    with open(f'{data_path}/test_subjects.txt', 'r') as f:
        test_dirs = f.read().splitlines()

    print(f'=> Naive model: {model_name}')
    if model_task == 'classification':
        _, y_train, subjects = get_baseline_datasets(train_dirs, coarse_targets,
                targets_only=True, config=config)
        _, y_val, subjects = get_baseline_datasets(val_dirs, coarse_targets,
             targets_only=True, config=config)
        _, y_test, subjects = get_baseline_datasets(test_dirs, coarse_targets,
             targets_only=True, config=config)
    elif model_task == 'regression':
        y_train, _, subjects = get_baseline_datasets(train_dirs, coarse_targets,
             targets_only=True, config=config)
        y_val, _, subjects = get_baseline_datasets(val_dirs, coarse_targets,
             targets_only=True, config=config)
        y_test, _, subjects = get_baseline_datasets(test_dirs, coarse_targets,
                targets_only=True, config=config)
    else:
        raise ValueError("Parameter 'model_task' must be one of " +
                "'classification' or 'regression'")

    # No validation set is needed
    y_train = np.hstack((y_train, y_val))

    # Get the mean and median
    mean = round(np.mean(y_train))
    median = round(np.median(y_train))

    if friedman:
        test_partitions_file = os.path.join(data_path,
                'test_partitions.txt')
        if not os.path.exists(test_partitions_file):
            raise FileNotFoundError(f"File note found: could not find " +
                    f"{test_partitions_file}.")

        with open(test_partitions_file, 'r') as f:
            partitions = f.read().splitlines()

        print(f'=> Predict mean')

        test_act = np.full(y_test.shape, mean)
        preds = {'Test seqs': subjects, 'True labels': y_test, 'Predictions': 
                test_act}
        df_pred = pd.DataFrame(preds)

        if model_task == 'regression':
            m = 'MAE'
            results = {'iters': 1000}
            
            results[m] = dict()
            results[m]['iters'] = []

            for k in tqdm(range(1000)):
                y_true = df_pred['True labels'].values
                y_true = resample(y_true, random_state=k)
                y_pred = df_pred['Predictions'].values
                y_pred = resample(y_pred, random_state=k)

                results[m]['iters'].append(calculate_metric(y_true, y_pred,
                    metric=m, verbose=False))

            iters = results[m]['iters']
            results[m]['mean'] = np.mean(iters)
            results[m]['median'] = np.median(iters)
            results[m]['std'] = np.std(iters)
            results[m]['2.5 percentile'] = np.percentile(iters, 2.5)
            results[m]['97.5 percentile'] = np.percentile(iters, 97.5)
            del results[m]['iters']
            print(results)

        print(f'=> Predict median')
        test_act = np.full(y_test.shape, median)
        preds = {'Test seqs': subjects, 'True labels': y_test, 'Predictions': 
                test_act}
        df_pred = pd.DataFrame(preds)

        if model_task == 'regression':
            metric = 'MAE'
            results = {'iters': 1000}
            
            results[m] = dict()
            results[m]['iters'] = []

            for k in tqdm(range(1000)):
                y_true = df_pred['True labels'].values
                y_true = resample(y_true, random_state=k)
                y_pred = df_pred['Predictions'].values
                y_pred = resample(y_pred, random_state=k)

                results[m]['iters'].append(calculate_metric(y_true, y_pred,
                    metric=m, verbose=False))

            iters = results[m]['iters']
            results[m]['mean'] = np.mean(iters)
            results[m]['median'] = np.median(iters)
            results[m]['std'] = np.std(iters)
            results[m]['2.5 percentile'] = np.percentile(iters, 2.5)
            results[m]['97.5 percentile'] = np.percentile(iters, 97.5)
            del results[m]['iters']
            print(results)

if __name__ == '__main__':
    main(parse_cl_args())

