#!/usr/bin/python3

"""logistic_regression.py

Script to create a Logistic Regression baseline.
"""

__author__ = "Bas Straathof"

import argparse, os, pickle

import numpy as np

from sys import argv

from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import make_scorer, cohen_kappa_score

from nicu_los.src.utils.modelling import get_baseline_datasets
from nicu_los.src.utils.evaluation import calculate_cohen_kappa, \
        evaluate_classification_model


def parse_cl_args():
    """Parses CL arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-sp', '--subjects-path', type=str,
            default='data', help='Path to the subjects directories.')
    parser.add_argument('-mp', '--models-path', type=str,
            default='models/logistic_regression/',
            help='Path to the models directory.')
    parser.add_argument('-mn', '--model-name', type=str, default="",
            help='Name of the model.')

    parser.add_argument('--grid-search', dest='grid_search',
            action='store_true')
    parser.add_argument('--no-grid-search', dest='grid_search',
            action='store_false')

    parser.add_argument('--coarse-targets', dest='coarse_targets',
            action='store_true')
    parser.add_argument('--fine-targets', dest='coarse_targets',
            action='store_false')

    parser.add_argument('--pre-imputed', dest='pre_imputed',
            action='store_true')
    parser.add_argument('--not-pre-imputed', dest='pre_imputed',
            action='store_false')

    parser.add_argument('--regularizer', type=str, default="l2",
            help='The regularizer: "l1" or "l2".')
    parser.add_argument('--C', type=float, default=1.0, help=('The Logistic ' \
            'Regression C parameter (i.e. float between 0.0 and 1.0).'))

    parser.add_argument('--training', dest='training', action='store_true')
    parser.add_argument('--testing', dest='training', action='store_false')

    parser.add_argument('--K', type=int, default=50, help=('How often to ' +
        'perform bootstrap sampling without replacement when evaluating ' +
        'the model'))
    parser.add_argument('--samples', type=int, default=16000, help=('Number ' +
    'of test samples per bootstrap'))

    parser.set_defaults(pre_imputed=False, grid_search=False,
            coarse_targets=True, training=True)

    return parser.parse_args(argv[1:])


def main(args):
    models_path = args.models_path
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    coarse_targets = args.coarse_targets
    data_path = args.subjects_path
    grid_search = args.grid_search
    model_name = args.model_name
    pre_imputed = args.pre_imputed
    regularizer = args.regularizer
    C = args.C

    if args.training:
        print(f'=> Training {model_name}')
    else:
        print(f'=> Evaluating {model_name}')

    with open(f'{data_path}/training_subjects.txt', 'r') as f:
        train_dirs = f.read().splitlines()
    with open(f'{data_path}/validation_subjects.txt', 'r') as f:
        val_dirs = f.read().splitlines()
    with open(f'{data_path}/test_subjects.txt', 'r') as f:
        test_dirs = f.read().splitlines()

    X_train, _, y_train = get_baseline_datasets(train_dirs, coarse_targets,
            pre_imputed)
    X_val, _, y_val = get_baseline_datasets(val_dirs, coarse_targets,
            pre_imputed)
    X_test, _, y_test = get_baseline_datasets(test_dirs, coarse_targets,
            pre_imputed)

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

    if args.training:
        print(f'=> Training {model_name}')
        print(f'=> Pre-imputed features: {pre_imputed}')
        print(f'=> Coarse targets: {coarse_targets}')
        print(f'=> Grid search: {grid_search}')

        # The training and validation set need to be fed conjointly to GridSearchCV
        X = np.vstack((X_train, X_val))
        y = np.hstack((y_train, y_val))

        # Define what indices of X belong to x_train, and which to x_Val
        val_idx = np.hstack((np.ones(X_train.shape[0])*-1, np.ones(X_val.shape[0])))
        ps = PredefinedSplit(test_fold=val_idx)

        # Define the parameter grid
        if grid_search:
            # Initialize the logistic regression hyper-paramters
            LR = LogisticRegression(random_state=42, multi_class="multinomial",
                    solver='saga')

            regularizers = ['l1', 'l2']
            Cs = [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001 ]
            print(f'=> Doing a grid search with the following regularizers and Cs.')
            print(f'=> Regularizers: {regularizers} .')
            print(f'=> Cs: {Cs} .')

            param_grid = dict(C=Cs, penalty=regularizers)

            # Initialize the grid serach using the predefined train-validation split
            clf = GridSearchCV(LR, param_grid=param_grid, n_jobs=8, cv=ps,
                    scoring=make_scorer(cohen_kappa_score), verbose=3)

            # Fit the GridSearchCV to find the optimal estimator
            print(f'=> Fitting the Logistic Regression model')
            clf.fit(X, y)

            # Extract the best estimator and fit again on all available training data
            print(f'=> Fitting the best Logistic Regression model on all available data')
            clf = clf.best_estimator_
            clf.fit(X, y)
        else:
            # Initialize the logistic regression estimator
            clf = LogisticRegression(random_state=42, penalty=regularizer, C=C,
                    multi_class="multinomial", solver='saga')

            print(f'=> Fitting Logistic Regression model with ' \
                    'regularizer={regularizer} and C={C}')
            clf.fit(X, y)

        # Predict on the training set
        train_preds = clf.predict_proba(X)
        train_act = np.argmax(train_preds, axis=1)

        # Predict on the testing set
        test_preds = clf.predict_proba(X_test)
        test_act = np.argmax(test_preds, axis=1)

        print('=> Evaluate fitted model on the training set')
        train_scores = evaluate_classification_model(y, train_act)

        print('=> Evaluate fitted model on the test set')
        test_scores = evaluate_classification_model(y_test, test_act)

        if model_name:
            print('=> Saving the model')
            f_name = os.path.join(models_path, f'results_{model_name}.txt')

            with open(f_name, "a") as f:
                if grid_search:
                    f.write(f'Best LR model: {clf.best_estimator_}:\n')
                else:
                    f.write(f'Best LR model: {clf.get_params}:\n')

                f.write(f'- Training scores:\n')
                for k, v in train_scores.items():
                    f.write(f'\t\t{k}: {v}\n')
                f.write(f'- Test scores:\n')
                for k, v in test_scores.items():
                    f.write(f'\t\t{k}: {v}\n')

            f_name = os.path.join(models_path, f'best_model_{model_name}.pkl')

            with open(f_name, 'wb') as f:
                pickle.dump(clf, f)

    else: # evaluation
        K = args.K
        samples = args.samples

        f_name = os.path.join(models_path, f'best_model_{model_name}.pkl')
        with open(f_name, 'rb') as f:
            clf = pickle.load(f)

        results_dir = os.path.join(models_path, 'results', model_name)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        f_name_results = os.path.join(results_dir, f'results.txt')

        print(f'=> Bootrapping evaluation (K={K}, samples={samples})')
        kappas = []
        for _ in range(K):
            indices = np.random.choice(X_test.shape[0], samples, replace=False)

            test_preds = clf.predict_proba(X_test[indices])
            test_preds = np.argmax(test_preds, axis=1)
        
            kappa = calculate_cohen_kappa(y_test[indices], test_preds,
                    verbose=False)

            kappas.append(kappa)

        mean_kappa = np.mean(kappas)
        std_kappa = np.std(kappas)
        print(f"Cohen's kappa:\n\tmean {mean_kappa}\n\tstd-dev {std_kappa}")

        with open(f_name_results, "a") as f:
            f.write(f'- Test scores K={K}, samples={samples}:\n')
            f.write(f"\tCohen's kappa mean: {mean_kappa}\n")
            f.write(f"\tCohen's kappa std-dev: {std_kappa}\n")
        

if __name__ == '__main__':
    main(parse_cl_args())

