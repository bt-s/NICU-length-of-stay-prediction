#!/usr/bin/python3

"""linear_regression.py

Script to create a Linear Regression baseline.
"""

__author__ = "Bas Straathof"

import argparse, csv, json, os, pickle

import numpy as np
import pandas as pd

from sys import argv
from tqdm import tqdm
from sklearn.utils import resample

from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from nicu_los.src.utils.data_helpers import get_baseline_datasets
from nicu_los.src.utils.evaluation import calculate_metric, \
        evaluate_regression_model

def parse_cl_args():
    """Parses CL arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-sp', '--subjects-path', type=str,
            default='data', help='Path to the subject directories.')
    parser.add_argument('-mp', '--model-path', type=str,
            default='models/linear_regression/',
            help='Path to the models directory.')
    parser.add_argument('-mn', '--model-name', type=str, default="",
            help='Name of the  model.')

    parser.add_argument('--pre-imputed', dest='pre_imputed',
            action='store_true')
    parser.add_argument('--not-pre-imputed', dest='pre_imputed',
            action='store_false')

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

    parser.set_defaults(pre_imputed=False, mode='training')

    return parser.parse_args(argv[1:])


def main(args):
    pre_imputed = args.pre_imputed
    data_path = args.subjects_path
    model_name = args.model_name
    model_path = args.model_path
    mode = args.mode
    config = args.config

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    if mode == 'training':
        print(f'=> Training {model_name}.')
    elif mode == 'prediction':
        print(f'=> Predicting {model_name}.')
    elif mode == 'evaluation':
        print(f'=> Evaluating {model_name}.')
    else:
        raise ValueError('Parameter "mode" must be one of: "training", ' +
                '"prediction", "evaluation".')

    print(f'=> Pre-imputed features: {pre_imputed}')

    if mode == 'training' or mode == 'prediction':
        with open(f'{data_path}/training_subjects.txt', 'r') as f:
            train_dirs = f.read().splitlines()
        with open(f'{data_path}/validation_subjects.txt', 'r') as f:
            val_dirs = f.read().splitlines()
        with open(f'{data_path}/test_subjects.txt', 'r') as f:
            test_dirs = f.read().splitlines()

        X_train, y_train, _ = get_baseline_datasets(train_dirs,
                pre_imputed=pre_imputed, config=config)
        X_val, y_val, _ = get_baseline_datasets(val_dirs, pre_imputed=pre_imputed,
                config=config)
        X_test, y_test, _ = get_baseline_datasets(test_dirs,
                pre_imputed=pre_imputed, config=config)

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

    if mode == 'training':
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
        f_name = os.path.join(model_path, f'results_{model_name}.txt')

        with open(f_name, "a") as f:
            f.write(f'LinearRegression:\n')
            f.write(f'- Train scores:\n')
            for k, v in train_scores.items():
                f.write(f'\t\t{k}: {v}\n')
            f.write(f'- Test scores:\n')
            for k, v in test_scores.items():
                f.write(f'\t\t{k}: {v}\n')

        # Save the model
        f_name = os.path.join(model_path, f'model_{model_name}.pkl')

        with open(f_name, 'wb') as f:
            pickle.dump(clf, f)

    elif mode == 'prediction':
        predictions_dir = os.path.join(model_path, 'predictions', model_name)
        if not os.path.exists(predictions_dir):
            os.makedirs(predictions_dir)
        f_name_predictions = os.path.join(predictions_dir, f'predictions.csv')

        f_clf = os.path.join(model_path, f'model_{model_name}.pkl')
        with open(f_clf, 'rb') as f:
            clf = pickle.load(f)

        y_pred = clf.predict(X_test)
        # Remaining LOS cannot be negative
        y_pred = np.maximum(y_pred, 0) 

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

        f_name = os.path.join(model_path, f'model_{model_name}.pkl')
        with open(f_name, 'rb') as f:
            clf = pickle.load(f)

        # Open the dataframe containing all predicitons on the test set
        df_pred = pd.read_csv(f_name_predictions, index_col=False)

        print(f'=> K={args.K} bootstrapping rounds')

        results = {'iters': args.K, 'MAE': {'iters': []}}

        for k in tqdm(range(args.K)):
            y_true = df_pred['True labels'].to_numpy()
            y_true = resample(y_true, random_state=k)
            y_pred = df_pred['Predictions'].to_numpy()
            y_pred = resample(y_pred, random_state=k)

            results['MAE']['iters'].append(calculate_metric(y_true, y_pred,
                metric='MAE', verbose=False))

        iters = results['MAE']['iters']
        results['MAE']['mean'] = np.mean(iters)
        results['MAE']['median'] = np.median(iters)
        results['MAE']['std'] = np.std(iters)
        results['MAE']['2.5 percentile'] = np.percentile(iters, 2.5)
        results['MAE']['97.5 percentile'] = np.percentile(iters, 97.5)
        del results['MAE']['iters']

        print(f'=> Writing results to {f_name_results}')
        with open(f_name_results, 'w') as f:
            json.dump(results, f)


if __name__ == '__main__':
    main(parse_cl_args())

