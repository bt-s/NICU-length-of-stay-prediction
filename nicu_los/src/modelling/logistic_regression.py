#!/usr/bin/python3

"""logistic_regression.py

Script to create a Logistic Regression baseline.
"""

__author__ = "Bas Straathof"

import argparse, os, pickle

import numpy as np

from sys import argv
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import make_scorer, cohen_kappa_score

from ..utils.evaluation_utils import evaluate_classification_model
from ..utils.modelling_utils import get_baseline_datasets


def parse_cl_args():
    """Parses CL arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-sp', '--subjects-path', type=str,
            default='data/',
            help='Path to the subjects directories.')
    parser.add_argument('-mp', '--models-path', type=str,
            default='models/logistic_regression/',
            help='Path to the models directory.')
    parser.add_argument('-pi', '--pre-imputed', type=int, default=0,
            help='Whether to use pre-imputed time-series.')
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

    data_path = args.subjects_path
    save_model = args.save_model
    pre_imputed = args.pre_imputed
    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    with open(f'{data_path}/training_subjects.txt', 'r') as f:
        train_dirs = f.read().splitlines()
    with open(f'{data_path}/validation_subjects.txt', 'r') as f:
        val_dirs = f.read().splitlines()
    with open(f'{data_path}/test_subjects.txt', 'r') as f:
        test_dirs = f.read().splitlines()

    # Get the data
    X_train, _, y_train = get_baseline_datasets(train_dirs)
    X_val, _, y_val = get_baseline_datasets(val_dirs)
    X_test, _, y_test = get_baseline_datasets(test_dirs)

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

    # The training and validation set need to be fed conjointly to GridSearchCV
    X = np.vstack((X_train, X_val))
    y = np.hstack((y_train, y_val))

    # Define what indices of X belong to x_train, and which to x_Val
    val_idx = np.hstack((np.ones(X_train.shape[0])*-1, np.ones(X_val.shape[0])))
    ps = PredefinedSplit(test_fold=val_idx)

    # Initialize the logistic regression hyper-paramters
    LR = LogisticRegression(random_state=42, multi_class="multinomial",
            solver='saga')

    # Define the parameter grid
    if args.grid_search:
        regularizers = ['l1', 'l2']
        Cs = [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001 ]
    else:
        regularizers = ['l2']
        Cs = [1.0]

    param_grid = dict(C=Cs, penalty=regularizers)

    # Initialize the grid serach using the predefined train-validation split
    clf = GridSearchCV(LR, param_grid=param_grid, n_jobs=-1, cv=ps,
            scoring=make_scorer(cohen_kappa_score), verbose=3)

    # Fit the GridSearchCV to find the optimal estimator
    clf.fit(X, y)

    # Extract the best estimator and fit again on all available training data
    best_clf = clf.best_estimator_
    best_clf.fit(X, y)

    # Predict on the training set
    train_preds = clf.predict_proba(X)
    train_act = np.argmax(train_preds, axis=1)

    # Predict on the testing set
    test_preds = clf.predict_proba(X_test)
    test_act = np.argmax(test_preds, axis=1)

    # Evaluate the model on the training set
    train_scores = evaluate_classification_model(y, train_act)

    # Evaluate the model on the test set
    test_scores = evaluate_classification_model(y_test, test_act)

    if save_model:
        f_name = os.path.join(args.models_path, f'results_{now}.txt')

        with open(f_name, "a") as f:
            f.write(f'Best LR model: {clf.best_estimator_}:\n')
            f.write(f'- Training scores:\n')
            for k, v in train_scores.items():
                f.write(f'\t\t{k}: {v}\n')
            f.write(f'- Test scores:\n')
            for k, v in test_scores.items():
                f.write(f'\t\t{k}: {v}\n')

        f_name = os.path.join(args.models_path,
                f'best_model_{now}.pkl')

        with open(f_name, 'wb') as f:
            pickle.dump(clf, f)


if __name__ == '__main__':
    main(parse_cl_args())

