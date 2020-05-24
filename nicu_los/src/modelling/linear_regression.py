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

from nicu_los.src.utils.modelling import get_baseline_datasets
from nicu_los.src.utils.evaluation import evaluate_regression_model


def parse_cl_args():
    """Parses CL arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-sp', '--subjects-path', type=str,
            default='data', help='Path to the subject directories.')
    parser.add_argument('-mp', '--models-path', type=str,
            default='models/linear_regression/',
            help='Path to the models directory.')
    parser.add_argument('-mn', '--model-name', type=str, default="",
            help='Name of the  model.')

    parser.add_argument('--pre-imputed', dest='pre_imputed',
            action='store_true')
    parser.add_argument('--not-pre-imputed', dest='pre_imputed',
            action='store_false')

    parser.add_argument('--training', dest='training', action='store_true')
    parser.add_argument('--testing', dest='training', action='store_false')

    parser.add_argument('--K', type=int, default=20, help=('How often to ' +
        'perform bootstrap sampling without replacement when evaluating ' +
        'the model'))
    parser.add_argument('--samples', type=int, default=16000, help=('Number ' +
    'of test samples per bootstrap'))

    parser.set_defaults(pre_imputed=False, training=True)

    return parser.parse_args(argv[1:])


def main(args):
    pre_imputed = args.pre_imputed
    data_path = args.subjects_path
    model_name = args.model_name
    models_path = args.models_path

    if not os.path.exists(models_path):
        os.makedirs(models_path)
    
    if args.training:
        print(f'=> Training {model_name}.')
    else:
        print(f'=> Evaluating {model_name}.')

    print(f'=> Pre-imputed features: {pre_imputed}')

    with open(f'{data_path}/training_subjects.txt', 'r') as f:
        train_dirs = f.read().splitlines()
    with open(f'{data_path}/validation_subjects.txt', 'r') as f:
        val_dirs = f.read().splitlines()
    with open(f'{data_path}/test_subjects.txt', 'r') as f:
        test_dirs = f.read().splitlines()

    X_train, y_train, _ = get_baseline_datasets(train_dirs,
            pre_imputed=pre_imputed)
    X_val, y_val, _ = get_baseline_datasets(val_dirs, pre_imputed=pre_imputed)
    X_test, y_test, _ = get_baseline_datasets(test_dirs,
            pre_imputed=pre_imputed)

    if not pre_imputed:
        print('=> Imputing missing data')
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean',
                fill_value='constant', verbose=0, copy=True)
        X_train = imputer.fit_transform(X_train)
        X_val = imputer.transform(X_val)
        X_test = imputer.transform(X_test)

    print('=> Normalizing the data')
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # No validation set is needed
    X_train = np.concatenate((X_train, X_val))
    y_train = np.hstack((y_train, y_val))

    if args.training:
        print(f'=> Fitting the Linear Regression model')
        clf = LinearRegression(n_jobs=-1)

        # Fit the model
        clf.fit(X_train, y_train)

        # Predict on the training set
        train_preds = clf.predict(X_train)
        # Remaining LOS cannot be negative
        train_preds = np.maximum(train_preds, 0)

        print('=> Evaluate fitted model on the training set')
        train_scores = evaluate_regression_model(y_train, train_preds)

        # Predict on the testing set
        test_preds = clf.predict(X_test)
        # Remaining LOS cannot be negative
        test_preds = np.maximum(test_preds, 0)

        print('=> Evaluate fitted model on the test set')
        test_scores = evaluate_regression_model(y_test, test_preds)

        print('=> Saving the model')
        f_name = os.path.join(models_path, f'results_{model_name}.txt')

        with open(f_name, "a") as f:
            f.write(f'LinearRegression:\n')
            f.write(f'- Train scores:\n')
            for k, v in train_scores.items():
                f.write(f'\t\t{k}: {v}\n')
            f.write(f'- Test scores:\n')
            for k, v in test_scores.items():
                f.write(f'\t\t{k}: {v}\n')

        # Save the model
        f_name = os.path.join(models_path, f'model_{model_name}.pkl')

        with open(f_name, 'wb') as f:
            pickle.dump(clf, f)

    else:
        f_name = os.path.join(models_path, f'model_{model_name}.pkl')
        with open(f_name, 'rb') as f:
            clf = pickle.load(f)

        print('=> Evaluate fitted model on bootstrap samples of the test set')
        MAEs = []
        for _ in range(args.K):
            indices = np.random.choice(X_test.shape[0], args.samples,
                    replace=False)
            test_preds = clf.predict(X_test[indices])
            # Remaining LOS cannot be negative
            test_preds = np.maximum(test_preds, 0) 

            test_scores = evaluate_regression_model(y_test[indices], test_preds,
                    verbose=False)
            MAEs.append(test_scores['mae'])

        mean_MAE = np.mean(MAEs)
        std_MAE = np.std(MAEs)
        print(f'Mean of MAE over {args.K} bootstrapping cycles of ' +
                f'{args.samples} samples: {mean_MAE}')
        print(f'Standard deviation of MAE over {args.K} bootstrapping cycles ' +
                f'of {args.samples} samples: {std_MAE}')


if __name__ == '__main__':
    main(parse_cl_args())

