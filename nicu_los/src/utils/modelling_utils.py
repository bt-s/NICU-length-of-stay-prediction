#!/usr/bin/python3

"""modelling_utils.py

Various utility functions for modelling
"""

__author__ = "Bas Straathof"

import errno, os
import numpy as np


def get_train_val_test_baseline_sets(data_path, task):
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), data_path)
    else:
        X_train = np.load(os.path.join(data_path,'X_train.npy'))
        y_train = np.load(os.path.join(data_path, 'y_train.npy'))
        t_train = np.load(os.path.join(data_path, 't_train.npy'))
        X_val = np.load(os.path.join(data_path,'X_val.npy'))
        y_val = np.load(os.path.join(data_path, 'y_val.npy'))
        t_val = np.load(os.path.join(data_path, 't_val.npy'))
        X_test = np.load(os.path.join(data_path,'X_test.npy'))
        y_test = np.load(os.path.join(data_path, 'y_test.npy'))
        t_test = np.load(os.path.join(data_path, 't_test.npy'))

    if task == 'classification':
        return X_train, t_train, X_val, t_val, X_test, t_test
    elif task == 'regression':
        return X_train, y_train, X_val, y_val, X_test, y_test
    else:
        raise ValueError("Task must be one of: 'classification', 'regression'.")


