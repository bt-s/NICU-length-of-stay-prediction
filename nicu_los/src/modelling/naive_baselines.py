#!/usr/bin/python3

"""naive_baselines.py

Script to test two naive baselines:
    (1) Always predicting the mean
    (2) Always predicting the median
"""

__author__ = "Bas Straathof"

import argparse, os, pickle

import numpy as np

from sys import argv
from datetime import datetime

from ..utils.evaluation_utils import evaluate_classification_model
from ..utils.modelling_utils import get_train_val_test_baseline_sets


def parse_cl_args():
    """Parses CL arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-sp', '--baseline-data-path', type=str,
            default='data/baseline_features/',
            help='Path to baseline features directories.')
    parser.add_argument('-mp', '--models-path', type=str,
            default='models/naive_baselines/',
            help='Path to the models directory.')
    parser.add_argument('-sm', '--save-model', type=bool, default=True,
            help='Whether to save the model.')
    parser.add_argument('-v', '--verbose', type=int,
            help='Level of verbosity in console output.', default=1)

    return parser.parse_args(argv[1:])


def main(args):
    v_print = print if args.verbose else lambda *a, **k: None

    if not os.path.exists(args.models_path):
        os.makedirs(args.models_path)
    data_path = args.baseline_data_path
    save_model = args.save_model
    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    v_print(f'Naive model: classification')
    # Get the targets
    _, y_train, _, y_val, _, y_test = \
            get_train_val_test_baseline_sets(data_path, task='classification')

    # No validation set is needed
    y_train = np.hstack((y_train, y_val))

    # Get the mean and median
    mean = round(np.mean(y_train))
    median = round(np.median(y_train))
    print(mean, median)

    v_print(f'- Predict mean')
    test_act = np.full(y_test.shape, mean)

    # Evaluate the model on the test set
    test_scores_mean = evaluate_classification_model(y_test, test_act)

    v_print(f'- Predict median')
    test_act = np.full(y_test.shape, median)

    # Evaluate the model on the test set
    test_scores_median = evaluate_classification_model(y_test, test_act)

    if save_model:
        f_name = os.path.join(args.models_path, f'results.txt')

        with open(f_name, "w") as f:
            f.write(f'Naive baselines:\n')
            f.write(f'- Predict mean:\n')
            for k, v in test_scores_mean.items():
                f.write(f'\t\t{k}: {v}\n')
            f.write(f'- Predict median:\n')
            for k, v in test_scores_median.items():
                f.write(f'\t\t{k}: {v}\n')

if __name__ == '__main__':
    main(parse_cl_args())

