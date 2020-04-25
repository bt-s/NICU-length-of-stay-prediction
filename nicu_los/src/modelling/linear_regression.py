#!/usr/bin/python3

"""linear_regression.py

Script to create a Linear Regression baseline.
"""

__author__ = "Bas Straathof"

import argparse, os, pickle

import numpy as np

from sys import argv
from datetime import datetime

from sklearn.linear_model import LinearRegression

from ..utils.evaluation_utils import evaluate_regression_model
from ..utils.modelling_utils import get_train_val_test_baseline_sets


def parse_cl_args():
    """Parses CL arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-sp', '--baseline-data-path', type=str,
            default='data/baseline_features/',
            help='Path to baseline features directories.')
    parser.add_argument('-mp', '--models-path', type=str,
            default='models/linear_regression/',
            help='Path to the models directory.')
    parser.add_argument('-sm', '--save-model', type=bool, default=True,
            help='Whether to save the model.')
    parser.add_argument('-gs', '--grid-search', type=bool, default=False,
            help='Whether to do a grid-search.')
    parser.add_argument('-v', '--verbose', type=int,
            help='Level of verbosity in console output.', default=1)

    return parser.parse_args(argv[1:])


def main(args):
    if not os.path.exists(args.models_path):
        os.makedirs(args.models_path)
    v_print = print if args.verbose else lambda *a, **k: None
    data_path = args.baseline_data_path
    save_model = args.save_model
    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    # Get the data
    X_train, y_train, X_val, y_val, X_test, y_test = \
            get_train_val_test_baseline_sets(data_path, task='regression')

    # No validation set is needed
    X_train = np.concatenate((X_train, X_val))
    y_train = np.hstack((y_train, y_val))

    v_print(f'Fitting the Linear Regression model')
    clf = LinearRegression(n_jobs=-1)

    # Fit the model
    clf.fit(X_train, y_train)

    # Predict on the training set
    train_preds = clf.predict(X_train)
    # Evaluate the model on the test set
    v_print('Evaluate on the training set')
    train_scores = evaluate_regression_model(y_train, train_preds)

    # Predict on the testing set
    test_preds = clf.predict(X_test)
    # Evaluate the model on the test set
    v_print('Evaluate on the test set')
    test_scores = evaluate_regression_model(y_test, test_preds)

    if save_model:
        # Save the results
        f_name = os.path.join(args.models_path, f'results_{now}.txt')

        with open(f_name, "a") as f:
            f.write(f'LinearRegression:\n')
            f.write(f'- Train scores:\n')
            for k, v in train_scores.items():
                f.write(f'\t\t{k}: {v}\n')
            f.write(f'- Test scores:\n')
            for k, v in test_scores.items():
                f.write(f'\t\t{k}: {v}\n')

        # Save the model
        f_name = os.path.join(args.models_path,
                f'model_{now}.pkl')

        with open(f_name, 'wb') as f:
            pickle.dump(clf, f)


if __name__ == '__main__':
    main(parse_cl_args())

