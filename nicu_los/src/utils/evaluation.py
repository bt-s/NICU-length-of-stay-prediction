#!/usr/bin/python3

"""evaluation.py

Various utility functions for model evaluation 
"""

__author__ = "Bas Straathof"


import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, cohen_kappa_score, \
        confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, \
        precision_score, recall_score, roc_auc_score


def calculate_metric(y_true, y_pred, metric, verbose=True):
    """Calculate a specifc classification metric

    Options:
        - accuracy
        - Cohen's kappa coefficient 
        - Recall 
        - Precision 
        - F1-score 
        - MAE (Mean Absolute Error)

    Args:
        y_true (list): True targets
        y_pred (list): Predicted targets
        verbose (bool): Whether to print the statistics

    Returns:
        res (float): Result of the requested metric
    """
    if metric == 'accuracy':
        res = accuracy_score(y_true, y_pred)
    elif metric == 'kappa':
        res = cohen_kappa_score(y_true, y_pred, weights='linear')
    elif metric == 'recall':
        res = recall_score(y_true, y_pred, average='weighted')
    elif metric == 'precision':
        res = precision_score(y_true, y_pred, average='weighted',
                zero_division=1)
    elif metric == 'f1':
        res = f1_score(y_true, y_pred, average='weighted')
    elif metric == 'MAE':
        res = mean_absolute_error(y_true, y_pred)
    else:
        raise ValueError('Invalid choice of metric.')

    return res 


def evaluate_classification_model(y_true, y_pred, verbose=True):
    """Function to collect various classification metrics

    Args:
        y_true (list): True targets
        y_pred (list): Predicted targets
        verbose (bool): Whether to print the statistics

    ReturnS:
        metrics (dict): Metrics, including the accuracy, Cohen's kappa 
                        coefficient , recall, precision, F1-score and
                        the confusion matrix
    """
    kappa = cohen_kappa_score(y_true, y_pred, weights='linear')
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)

    if verbose:
        print(f'=> Accuracy: {acc}')
        print(f'=> Linear Cohen Kappa Score: {kappa}')
        print(f'=> Precision: {precision}')
        print(f'=> Recall: {recall}')
        print(f'=> F1: {f1}')
        print(f'=> Confusion matrix:\n{cm}')

    metrics = {"accuracy": acc, 'kappa': kappa, 'precision': precision,
            'recall': recall, 'f1': f1, 'cm': cm}

    return metrics


def evaluate_regression_model(y_true, y_pred, verbose=True):
    """Function to collect various regression metrics

    Args:
        y_true (list): True targets
        y_pred (list): Predicted targets
        verbose (bool): Whether to print the statistics

    Returns:
        metrics (dict): Metrics, including the mean absolute error, 
                        the mean squared error, the root mean squared error,
                        and the mean absolute percentage error.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_perc_error(y_true, y_pred)

    if verbose:
        print(f'=> Mean Absolute Error (MAE): {mae}')
        print(f'=> Mean Squared Error (MSE): {mse}')
        print(f'=> Root Mean Squared Error (RMSE): {rmse}')
        print(f'=> Mean Aboslute Perentage Error (MAPE): {mape}')

    metrics = {'mae': mae, 'mse': mse, 'rmse': rmse, 'mape': mape}

    return metrics


def calculate_mean_absolute_error(y_true, y_pred, verbose=True):
    """Function to calculate the mean absolute error
    
    Args:
        y_true (list): True targets
        y_pred (list): Predicted targets
        verbose (bool): Whether to print the statistics

    Returns:
        mae (float): The mean absolute error
    """
    mae = mean_absolute_error(y_true, y_pred)

    if verbose:
        print(f'=> Mean Absolute Error (MAE): {mae}')

    return mae


def calculate_cohen_kappa(y_true, y_pred, verbose=True):
    """Function to calculate Cohen's kappa statistic 
    
    Args:
        y_true (list): True targets
        y_pred (list): Predicted targets
        verbose (bool): Whether to print the statistics

    Returns:
        kappa (float): The linearly weighted Cohen kappa coefficient 
    """
    kappa = cohen_kappa_score(y_true, y_pred, weights='linear')

    if verbose:
        print(f'=> Linear Cohen Kappa Score: {kappa}')

    return kappa


def calculate_confusion_matrix(y_true, y_pred, normalize=None):
    return confusion_matrix(y_true, y_pred, normalize=normalize)

