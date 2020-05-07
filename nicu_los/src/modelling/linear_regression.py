#!/usr/bin/python3

"""linear_regression.py

Script to create a Linear Regression baseline.
"""

__author__ = "Bas Straathof"

import argparse, os, pickle

import numpy as np

from sys import argv

from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from nicu_los.src.utils.modelling_utils import evaluate_regression_model, \
        get_baseline_datasets


def parse_cl_args():
    """Parses CL arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-sp', '--subjects-path', type=str,
            default='data', help='Path to the subject directories.')
    parser.add_argument('-mp', '--models-path', type=str,
            default='models/linear_regression/',
            help='Path to the models directory.')
    parser.add_argument('-pi', '--pre-imputed', type=int, default=0,
            help='Whether to use pre-imputed time-series.')
    parser.add_argument('-mn', '--model-name', type=str, default="",
            help='Name of the  model.')

    return parser.parse_args(argv[1:])


def main(args):
    if not os.path.exists(args.models_path):
        os.makedirs(args.models_path)
    pre_imputed = args.pre_imputed
    data_path = args.subjects_path
    model_name = args.model_name

    with open(f'{data_path}/training_subjects.txt', 'r') as f:
        train_dirs = f.read().splitlines()
    with open(f'{data_path}/validation_subjects.txt', 'r') as f:
        val_dirs = f.read().splitlines()
    with open(f'{data_path}/test_subjects.txt', 'r') as f:
        test_dirs = f.read().splitlines()

    # Get the data
    X_train, y_train, _ = get_baseline_datasets(train_dirs)
    X_val, y_val, _ = get_baseline_datasets(val_dirs)
    X_test, y_test, _ = get_baseline_datasets(test_dirs)

    if not pre_imputed:
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean',
                fill_value='constant', verbose=0, copy=True)
        X_train = imputer.fit_transform(X_train)
        X_val = imputer.transform(X_val)
        X_test = imputer.transform(X_test)

    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # No validation set is needed
    X_train = np.concatenate((X_train, X_val))
    y_train = np.hstack((y_train, y_val))

    print(f'Fitting the Linear Regression model')
    clf = LinearRegression(n_jobs=-1)

    # Fit the model
    clf.fit(X_train, y_train)

    # Predict on the training set
    train_preds = clf.predict(X_train)
    train_preds = np.maximum(train_preds, np.min(y_train))
    # Evaluate the model on the test set
    print('Evaluate on the training set')
    train_scores = evaluate_regression_model(y_train, train_preds)

    # Predict on the testing set
    test_preds = clf.predict(X_test)
    test_preds = np.maximum(test_preds, np.min(y_train))

    # Evaluate the model on the test set
    print('Evaluate on the test set')
    test_scores = evaluate_regression_model(y_test, test_preds)

    if save_model:
        # Save the results
        f_name = os.path.join(args.models_path, f'results_{model_name}.txt')

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
                f'model_{model_name}.pkl')

        with open(f_name, 'wb') as f:
            pickle.dump(clf, f)


if __name__ == '__main__':
    main(parse_cl_args())

