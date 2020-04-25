#!/usr/bin/python3

"""modelling_utils.py

Various utility functions for modelling
"""

__author__ = "Bas Straathof"

import errno, os
import numpy as np


def get_train_val_test_baseline_sets(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), data_path)
    else:
        X_train = np.load(os.path.join(data_path,'X_train.npy'))
        y_train = np.load(os.path.join(data_path, 't_train.npy'))
        X_val = np.load(os.path.join(data_path,'X_val.npy'))
        y_val = np.load(os.path.join(data_path, 't_val.npy'))
        X_test = np.load(os.path.join(data_path,'X_test.npy'))
        y_test = np.load(os.path.join(data_path, 't_test.npy'))

    return X_train, y_train, X_val, y_val, X_test, y_test

