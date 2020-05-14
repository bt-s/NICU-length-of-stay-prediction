#!/usr/bin/python3

"""naive_baselines.py

Script to test two naive baselines:
    (1) Always predict the mean
    (2) Always predict the median
"""

__author__ = "Bas Straathof"

import argparse, os, pickle

import numpy as np

from sys import argv
from datetime import datetime

from nicu_los.src.utils.modelling import evaluate_classification_model, \
        evaluate_regression_model, get_baseline_datasets


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
    parser.add_argument('--no-coarse-targets', dest='coarse_targets',
            action='store_false')

    parser.set_defaults(coarse_targets=True)

    return parser.parse_args(argv[1:])


def main(args):
    if not os.path.exists(args.models_path):
        os.makedirs(args.models_path)

    data_path = args.subjects_path
    model_name = args.model_name
    model_task = args.model_task
    coarse_targets = args.coarse_targets

    with open(f'{data_path}/training_subjects.txt', 'r') as f:
        train_dirs = f.read().splitlines()
    with open(f'{data_path}/validation_subjects.txt', 'r') as f:
        val_dirs = f.read().splitlines()
    with open(f'{data_path}/test_subjects.txt', 'r') as f:
        test_dirs = f.read().splitlines()

    print(f'=> Naive model: {model_name}')
    if model_task == 'classification':
        _, _, y_train = get_baseline_datasets(train_dirs, coarse_targets,
                targets_only=True)
        _, _, y_val = get_baseline_datasets(val_dirs, coarse_targets,
                targets_only=True)
        _, _, y_test = get_baseline_datasets(test_dirs, coarse_targets,
                targets_only=True)
    elif model_task == 'regression':
        _, y_train, _ = get_baseline_datasets(train_dirs, coarse_targets,
                targets_only=True)
        _, y_val, _ = get_baseline_datasets(val_dirs, coarse_targets,
                targets_only=True)
        _, y_test, _ = get_baseline_datasets(test_dirs, coarse_targets,
                targets_only=True)
    else:
        raise ValueError("Parameter 'model_task' must be one of " +
                "'classification' or 'regression'")

    # No validation set is needed
    y_train = np.hstack((y_train, y_val))

    # Get the mean and median
    mean = round(np.mean(y_train))
    median = round(np.median(y_train))

    print(f'=> Predict mean')
    test_act = np.full(y_test.shape, mean)

    # Evaluate the model on the test set
    if model_task == 'classification':
        test_scores_mean = evaluate_classification_model(y_test, test_act)
    else:
        test_scores_mean = evaluate_regression_model(y_test, test_act)

    print(f'=> Predict median')
    test_act = np.full(y_test.shape, median)

    # Evaluate the model on the test set
    if model_task == 'classification':
        test_scores_median = evaluate_classification_model(y_test, test_act)
    else:
        test_scores_median = evaluate_regression_model(y_test, test_act)

    if model_name:
        print('=> Saving the model')
        f_name = os.path.join(args.models_path, f'results_{model_name}.txt')

        with open(f_name, "w") as f:
            f.write(f'Naive baseline:\n')
            f.write(f'- Predict mean:\n')
            for k, v in test_scores_mean.items():
                f.write(f'\t\t{k}: {v}\n')
            f.write(f'- Predict median:\n')
            for k, v in test_scores_median.items():
                f.write(f'\t\t{k}: {v}\n')

if __name__ == '__main__':
    main(parse_cl_args())

