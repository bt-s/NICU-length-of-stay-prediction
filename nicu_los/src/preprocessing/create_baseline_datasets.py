#!/usr/bin/python3

"""create_baseline_datasets.py

The baselne models are not capable of modelling sequential data. For these
models, hand-crafted features are obtained from the timeseries.
"""

__author__ = "Bas Straathof"

import argparse, json, os

from sys import argv

import numpy as np
import multiprocessing as mp
from itertools import repeat

from scipy.stats import skew
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from nicu_los.src.utils.preprocessing_utils import split_train_val
from nicu_los.src.utils.utils import get_subject_dirs, remove_subject_dir


def parse_cl_args():
    """Parses CL arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-sp', '--subjects-path', type=str, default='data/',
            help='Path to subject directories.')
    parser.add_argument('-pi', '--pre-imputed', type=int, default=0,
            help='Whether to use pre-imputed time-series.')

    return parser.parse_args(argv[1:])


def get_first(s):
    """Get the first item in a Pandas Series"""
    return s.iloc[0]


def get_last(s):
    """Get the last item in a Pandas Series"""
    return s.iloc[-1]


def get_subseq(df, perc_start, perc_end):
    """Get a subsequence from a dataframe

    Args:
        df (pd.DataFrame): Pandas DataFrame
        perc_start (int): Starting percentage of the subsequence
        perc_end (int): Ending percentage of the subsequence

    Returns:
        subseq (pd.DataFrame): The requested subsequence
    """
    start = int(len(df) * perc_start/100)
    end = int(len(df) * perc_end/100)

    df = df.iloc[start:end]

    return df


def compute_stats(subseq, stat_fns):
    """Compute the statistics for a subsequence

    Args:
        subseq (pd.DataFrame): Time series subsequence
        stat_fns (list): List of statistical functions

    Returns:
        stats (np.array): Array of statistics computed over the subsequence
    """
    stats = np.array([subseq.apply(fn, axis=0) for fn in stat_fns],
            dtype=np.float32).flatten()

    return stats


def compute_stats_for_subseqs(timeseries, variables, stat_fns, subseqs=[]):
    """Obtain statistics per subsequence

    Args:
        timeseries (pd.DataFrame): Pandas DataFrame containing the time series
        variables (list): List of variables
        stat_fns (list): List of statistical functions
        subseqs (list): List of subsequences

    Returns:
        X (np.ndarray): The statistics of the subsequences
    """
    # Shape: # of subseqs, # of funcs times # of variables
    X = np.zeros((len(subseqs), len(stat_fns)*len(variables)))

    for i, (start_perc, end_perc) in enumerate(subseqs):
        subseq = get_subseq(timeseries, start_perc, end_perc)
        # No statistics can be computed from an empty dataframe
        if not subseq.empty:
            X[i, :] = compute_stats(subseq.loc[:, variables], stat_fns)

    X = X.flatten()

    return X


def create_baseline_datasets_per_subject(subject_dir, variables, stat_fns,
        subseqs, pre_imputed=False):
    """Create baseline data sets

    Args:
        subject_dir (str): Subject directory
        variables (list): List of variables
        stat_fns (list): List of statistical functions
        subseqs (list): List of subsequences
        pre_imputed (bool): Whether to use pre-imputed time series
    """
    if pre_imputed:
        ts = pd.read_csv(os.path.join(subject_dir, 'timeseries_imputed.csv'))
    else:
        ts = pd.read_csv(os.path.join(subject_dir, 'timeseries.csv'))

    y = ts.LOS_HOURS.to_numpy()
    t = ts.TARGET.to_numpy()

    X = np.zeros((len(ts), len(stat_fns)*len(subseqs)*len(variables)))

    for i in range(1, len(ts)):
        X[i] = compute_stats_for_subseqs(ts[0:i], variables, stat_fns, subseqs)

    np.save(f'{subject_dir}/X_baseline', X)
    np.save(f'{subject_dir}/y_baseline', y)
    np.save(f'{subject_dir}/t_baseline', t)


def split_train_val(train_dirs_path, val_perc=20):
    """Split the training data in a training and validation set

    Args:
        train_dirs_path (str): Path to the training directories
        val_perc (int): Percentage of data to be reserved for validation

    Returns:
        train_dirs (list): List of training directories
        val_dirs (list): List of validation directories
    """
    train_dirs = get_subject_dirs(train_dirs_path)

    # Get two arrays: one of training targets and one of the
    # corresponding subjects
    targets_train = np.zeros(len(train_dirs))
    subjects_train = np.zeros(len(train_dirs))
    for i, sd in enumerate(train_dirs):
        df_ts = pd.read_csv(os.path.join(sd, 'timeseries.csv'))
        targets_train[i] = df_ts.TARGET.iloc[0]
        subject_id = [int(s) for s in sd.split('/') if s.isdigit()][-1]
        subjects_train[i] = subject_id

    # Split the subjects list into training subjects list and a
    # validation subjects list, in a stratified manner
    subjects_train, subjects_val, _, _ = train_test_split(
            subjects_train, targets_train, test_size=val_perc/100,
            stratify=targets_train, shuffle=True)

    train_dirs = [f'{train_dirs_path}/{int(i)}' for i in subjects_train]
    val_dirs = [f'{train_dirs_path}/{int(i)}' for i in subjects_val]

    return train_dirs, val_dirs


def main(args):
    subjects_path = args.subjects_path

    train_sub_path = os.path.join(subjects_path, 'training_subjects.txt')
    val_sub_path = os.path.join(subjects_path, 'validation_subjects.txt')
    test_sub_path = os.path.join(subjects_path, 'test_subjects.txt')

    if os.path.exists(train_sub_path) and os.path.exists(val_sub_path) \
            and os.path.exists(test_sub_path):
        with open(train_sub_path, 'r') as f:
            train_dirs = f.read().splitlines()
        with open(val_sub_path, 'r') as f:
            val_dirs = f.read().splitlines()
        with open(test_sub_path, 'r') as f:
            test_dirs = f.read().splitlines()
    else:
        test_dirs = get_subject_dirs(os.path.join(subjects_path, 'test/'))
        train_dirs, val_dirs = split_train_val(os.path.join(subjects_path,
            'train/'), val_perc=0.2)

        # Write the training, validation and test subjects to files
        with open(train_sub_path,'w') as f: f.write('\n'.join(train_dirs))
        with open(val_sub_path,'w') as f: f.write('\n'.join(val_dirs))
        with open(test_sub_path,'w') as f: f.write('\n'.join(test_dirs))

    with open('nicu_los/config.json') as f:
        config = json.load(f)
        variables = config['variables']
        subseqs = config['baseline_subsequences']
        stat_fns = config['stat_fns']
        stat_fns = [eval(x) for x in stat_fns]

    subject_dirs = train_dirs + val_dirs + test_dirs

    with mp.Pool() as pool:
        for _ in tqdm(pool.istarmap(create_baseline_datasets_per_subject,
            zip(subject_dirs, repeat(variables), repeat(stat_fns),
                repeat(subseqs), repeat(args.pre_imputed)))):
            pass


if __name__ == '__main__':
    main(parse_cl_args())

