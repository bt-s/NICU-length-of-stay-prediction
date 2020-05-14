#!/usr/bin/python3

"""evaluation.py

Various utility functions for model evaluation 
"""

__author__ = "Bas Straathof"


import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, cohen_kappa_score, \
        confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, \
        plot_confusion_matrix, precision_score, recall_score, roc_auc_score


def mean_absolute_perc_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 0.1))) * 100


def evaluate_classification_model(y_true, y_pred, verbose=1):
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
        print(f'=> Confusion matrix:\n{cm}')

    return {"accuracy": acc, 'kappa': kappa, 'precision': precision,
            'recall': recall, 'cm': cm}


def evaluate_regression_model(y_true, y_pred, verbose=1):
    mad = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_perc_error(y_true, y_pred)

    if verbose:
        print(f'=> Mean Absolute Deviation (MAD): {mad}')
        print(f'=> Mean Squared Error (MSE): {mse}')
        print(f'=> Root Mean Squared Error (RMSE): {rmse}')
        print(f'=> Mean Aboslute Perentage Error (MAPE): {mape}')

    return {'mad': mad, 'mse': mse, 'rmse': rmse, 'mape': mape}


def get_confusion_matrix(model, X, y, save_plot='', class_names=['0-1',
    '1-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8', '8-14', '14+']):
    titles_options = [("Confusion matrix, without normalization", None),
                    ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(model, X, y,
                                    display_labels=class_names,
                                    cmap=plt.cm.Blues,
                                    normalize=normalize)
        disp.ax_.set_title(title)

        if save_plot:
            if normalize: save_plot += '_normalized'
            plt.savefig(save_plot, format="pdf", bbox_inches='tight',
                    pad_inches=0)
            plt.close()
        else:
            plt.show()
            plt.close()

