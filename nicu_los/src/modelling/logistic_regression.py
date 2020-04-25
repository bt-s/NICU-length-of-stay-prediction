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

from ..utils.evaluation_utils import evaluate_classification_model
from ..utils.modelling_utils import get_train_val_test_baseline_sets


def parse_cl_args():
    """Parses CL arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-sp', '--baseline-data-path', type=str,
            default='data/baseline_features/',
            help='Path to baseline features directories.')
    parser.add_argument('-mp', '--models-path', type=str,
            default='models/logistic_regression/',
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
            get_train_val_test_baseline_sets(data_path, task='classification')

    if args.grid_search:
        regularizers = ['l1', 'l1', 'l1', 'l1', 'l1', 'l2', 'l2', 'l2', 'l2',
                'l2']
        Cs = [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 1.0, 0.1, 0.01, 0.001,
                0.0001, 0.00001]
    else:
        regularizers = ['l2']
        Cs = [1.0]

    best_model, best_kappa = None, 0
    for (regularizer, C) in zip(regularizers, Cs):
        v_print(f'Fitting LR model with penalty={regularizer} and C={C}')
        # Create Numpy arrays for storing the activations
        train_act = np.zeros(shape=y_train.shape, dtype=float)
        val_act = np.zeros(shape=y_val.shape, dtype=float)
        test_act = np.zeros(shape=y_test.shape, dtype=float)

        clf = LogisticRegression(random_state=42, multi_class="multinomial",
                penalty=regularizer, C=C, solver='saga', n_jobs=-1)

        # Fit the model
        clf.fit(X_train, y_train)

        # Predict on the validation set
        val_preds = clf.predict_proba(X_val)
        val_act[:] = np.argmax(val_preds, axis=1)

        # Predict on the testing set
        test_preds = clf.predict_proba(X_test)
        test_act[:] = np.argmax(test_preds, axis=1)

        # Evaluate the model on the validation set
        val_scores = evaluate_classification_model(y_val, val_act)

        # Only keep the best model
        if val_scores['kappa'] > best_kappa:
            best_kappa = val_scores['kappa']
            best_model = clf

        # Evaluate the model on the test set
        test_scores = evaluate_classification_model(y_test, test_act)

        if save_model:
            f_name = os.path.join(args.models_path, f'results_{now}.txt')

            with open(f_name, "a") as f:
                f.write(f'LR: penalty={regularizer}, C={C}:\n')
                f.write(f'- Validation scores:\n')
                for k, v in val_scores.items():
                    f.write(f'\t\t{k}: {v}\n')
                f.write(f'- Test scores:\n')
                for k, v in test_scores.items():
                    f.write(f'\t\t{k}: {v}\n')

    if save_model:
        f_name = os.path.join(args.models_path,
                f'model_{now}_{regularizer}_{C}_kappa_{best_kappa}.pkl')

        with open(f_name, 'wb') as f:
            pickle.dump(clf, f)


if __name__ == '__main__':
    main(parse_cl_args())

