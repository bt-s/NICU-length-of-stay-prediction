#!/usr/bin/python3

"""logistic_regression.py

Script to create a Logistic Regression baseline.
"""

__author__ = "Bas Straathof"

import argparse, csv, json, os, pickle

import numpy as np
import pandas as pd

from sys import argv
from tqdm import tqdm

from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import make_scorer, cohen_kappa_score

from nicu_los.src.utils.data_helpers import get_baseline_datasets
from nicu_los.src.utils.evaluation import calculate_metric, \
        calculate_confusion_matrix, evaluate_classification_model
from nicu_los.src.utils.visualization import plot_confusion_matrix 


def parse_cl_args():
    """Parses CL arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-sp', '--subjects-path', type=str,
            default='data', help='Path to the subjects directories.')
    parser.add_argument('-mp', '--model-path', type=str,
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

    parser.add_argument('--training', dest='mode', action='store_const',
            const='training')
    parser.add_argument('--prediction', dest='mode', action='store_const',
            const='prediction')
    parser.add_argument('--evaluation', dest='mode', action='store_const',
            const='evaluation')

    parser.add_argument('--K', type=int, default=1000, help=('How often to ' +
        'perform bootstrap sampling without replacement when evaluating ' +
        'the model'))

    parser.add_argument('-c', '--config', type=str,
            default='nicu_los/config.json', help='Path to the config file')

    parser.set_defaults(pre_imputed=False, grid_search=False,
            coarse_targets=True, mode='training')

    return parser.parse_args(argv[1:])


def main(args):
    model_path = args.model_path
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    coarse_targets = args.coarse_targets
    data_path = args.subjects_path
    grid_search = args.grid_search
    model_name = args.model_name
    pre_imputed = args.pre_imputed
    regularizer = args.regularizer
    C = args.C
    mode = args.mode
    config = args.config

    if coarse_targets:
        output_dimension = 3
    else:
        output_dimension = 10

    if mode == 'training':
        print(f'=> Training {model_name}')
    elif mode == 'prediction':
        print(f'=> Predicting {model_name}.')
    elif mode == 'evaluation':
        print(f'=> Evaluating {model_name}.')
    else:
        raise ValueError('Parameter "mode" must be one of: "training", ' +
                '"prediction", "evaluation".')

    if mode == 'training' or mode == 'prediction':
        with open(f'{data_path}/training_subjects.txt', 'r') as f:
            train_dirs = f.read().splitlines()
        with open(f'{data_path}/validation_subjects.txt', 'r') as f:
            val_dirs = f.read().splitlines()
        with open(f'{data_path}/test_subjects.txt', 'r') as f:
            test_dirs = f.read().splitlines()

        X_train, _, y_train = get_baseline_datasets(train_dirs, coarse_targets,
                pre_imputed, config=config)
        X_val, _, y_val = get_baseline_datasets(val_dirs, coarse_targets,
                pre_imputed, config=config)
        X_test, _, y_test = get_baseline_datasets(test_dirs, coarse_targets,
                pre_imputed, config=config)

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

    if mode == 'training':
        print(f'=> Training {model_name}')
        print(f'=> Pre-imputed features: {pre_imputed}')
        print(f'=> Coarse targets: {coarse_targets}')
        print(f'=> Grid search: {grid_search}')

        # The training and validation set need to be fed conjointly to
        # GridSearchCV
        X = np.vstack((X_train, X_val))
        y = np.hstack((y_train, y_val))

        # Define what indices of X belong to x_train, and which to x_Val
        val_idx = np.hstack((np.ones(X_train.shape[0])*-1, 
            np.ones(X_val.shape[0])))
        ps = PredefinedSplit(test_fold=val_idx)

        # Define the parameter grid
        if grid_search:
            # Initialize the logistic regression hyper-paramters
            LR = LogisticRegression(random_state=42, multi_class="multinomial",
                    solver='saga')

            regularizers = ['l1', 'l2']
            Cs = [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001 ]
            print(f'=> Doing a grid search with the following regularizers ' +
                    'and Cs.')
            print(f'=> Regularizers: {regularizers} .')
            print(f'=> Cs: {Cs} .')

            param_grid = dict(C=Cs, penalty=regularizers)

            # Initialize the grid serach using the predefined train-validation 
            # split
            clf = GridSearchCV(LR, param_grid=param_grid, n_jobs=6, cv=ps,
                    scoring=make_scorer(cohen_kappa_score), verbose=3)

            # Fit the GridSearchCV to find the optimal estimator
            print(f'=> Fitting the Logistic Regression model')
            clf.fit(X, y)
        else:
            # Initialize the logistic regression estimator
            clf = LogisticRegression(random_state=42, penalty=regularizer, C=C,
                    multi_class="multinomial", solver='saga')

            print(f'=> Fitting Logistic Regression model with ' \
                    f'regularizer={regularizer} and C={C}')
            clf.fit(X_train, y_train)

        # Predict on the training set
        train_preds = clf.predict_proba(X_train)
        train_act = np.argmax(train_preds, axis=1)

        # Predict on the testing set
        test_preds = clf.predict_proba(X_test)
        test_act = np.argmax(test_preds, axis=1)

        print('=> Evaluate fitted model on the training set')
        train_scores = evaluate_classification_model(y_train, train_act)

        print('=> Evaluate fitted model on the test set')
        test_scores = evaluate_classification_model(y_test, test_act)

        if model_name:
            print('=> Saving the model')
            f_name = os.path.join(model_path, f'results_{model_name}.txt')

            with open(f_name, "a") as f:
                f.write(f'Best LR model: {clf.get_params}:\n')
                f.write(f'- Training scores:\n')
                for k, v in train_scores.items():
                    f.write(f'\t\t{k}: {v}\n')
                f.write(f'- Test scores:\n')
                for k, v in test_scores.items():
                    f.write(f'\t\t{k}: {v}\n')

            f_name = os.path.join(model_path, f'best_model_{model_name}.pkl')

            with open(f_name, 'wb') as f:
                pickle.dump(clf, f)

    elif mode == 'prediction':
        predictions_dir = os.path.join(model_path, 'predictions', model_name)
        if not os.path.exists(predictions_dir):
            os.makedirs(predictions_dir)
        f_name_predictions = os.path.join(predictions_dir, f'predictions.csv')

        f_clf = os.path.join(model_path, f'best_model_{model_name}.pkl')
        with open(f_clf, 'rb') as f:
            clf = pickle.load(f)

        y_pred = clf.predict_proba(X_test)
        y_pred = np.argmax(y_pred, axis=1)

        print(f'=> Writing results to {f_name_predictions}')
        with open(f_name_predictions, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['True labels', 'Predictions'])
            writer.writerows(zip(y_test, y_pred))

    elif mode == 'evaluation':
        f_name_predictions = os.path.join(model_path, 'predictions',
                model_name, 'predictions.csv')
        if not os.path.exists(f_name_predictions):
            raise FileNotFoundError("File note found: make sure to predict " +
                    "first.")

        results_dir = os.path.join(model_path, 'results', model_name)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        f_name_results = os.path.join(results_dir, f'results.json')
        f_name_confusion_matrix = os.path.join(results_dir, f'cm.pdf')
        f_name_confusion_matrix_normalized = os.path.join(results_dir,
                f'cm_normalized.pdf')

        f_name = os.path.join(model_path, f'best_model_{model_name}.pkl')
        with open(f_name, 'rb') as f:
            clf = pickle.load(f)

        # Open the dataframe containing all predicitons on the test set
        df_pred = pd.read_csv(f_name_predictions, index_col=False)

        print(f'=> K={args.K} bootstrapping rounds')

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

        # Create and plot confusion matrix
        y_true = df_pred['True labels'].to_numpy()
        y_pred = df_pred['Predictions'].to_numpy()

        cm = calculate_confusion_matrix(y_true, y_pred)
        cm_normalized = calculate_confusion_matrix(y_true, y_pred,
                normalize='pred')
        
        plot_confusion_matrix(cm, output_dimension, f_name_confusion_matrix)
        plot_confusion_matrix(cm_normalized, output_dimension, 
                f_name_confusion_matrix_normalized)

        results['confusion matrix'] = cm.tolist()
        results['confusion matrix normalized'] = cm_normalized.tolist()

        print(f'=> Writing results to {f_name_results}')
        with open(f_name_results, 'w') as f:
            json.dump(results, f)


if __name__ == '__main__':
    main(parse_cl_args())

