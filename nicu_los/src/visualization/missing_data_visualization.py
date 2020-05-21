#!/usr/bin/python3

"""missing_data_visualization.py

Script to visualize the missing data per variable.
"""

__author__ = "Bas Straathof"

import argparse, errno, os, json

import pandas as pd
import numpy as np
import missingno

from sys import argv
from tqdm import tqdm

from nicu_los.src.utils.utils import get_subject_dirs


def parse_cl_args():
    """Parses CL arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-sp', '--subjects-path', type=str, default='data',
            help='Path to subject directories.')
    parser.add_argument('-pp', '--plots-path', type=str, default='plots/',
            help='Path to plots directory.')

    return parser.parse_args(argv[1:])


def main(args):
    train_path = os.path.join(args.subjects_path, 'train')
    test_path = os.path.join(args.subjects_path, 'test')

    if not (os.path.exists(train_path) or os.path.exists(test_path)):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), train_path)
    elif not (os.path.exists(args.plots_path)): os.makedirs(args.plots_path)

    subject_directories_train = get_subject_dirs(train_path)
    subject_directories_test = get_subject_dirs(test_path)
    subject_directories = subject_directories_train + subject_directories_test

    los_hours, los_remaining_hours, los_targets_coarse, \
            los_remaining_targets_coarse, los_targets_fine, \
            los_remaining_targets_fine =  [], [], [], [], [], []

    with open('nicu_los/config.json') as f:
        config = json.load(f)
        variables = config['variables']

    # Store all data in a single dataframe
    complete_data_df = pd.DataFrame(columns=variables)
    # Per subject, store which variables have no values in the time series
    subject_no_values_df = pd.DataFrame(columns=variables)
    for i, sd in enumerate(tqdm(subject_directories)):
        ts = pd.read_csv(os.path.join(sd, 'timeseries.csv'))
        ts = ts[variables]

        empty_vars_series = ts.notnull().any()
        subject_no_values_df = subject_no_values_df.append( empty_vars_series,
                ignore_index=True)
        complete_data_df = complete_data_df.append(ts)

    # Visualize the percentage of missing values per variable for all data 
    ax = missingno.bar(complete_data_df, color=(31/256, 119/256, 180/256))
    ax.figure.savefig(os.path.join(args.plots_path,
        'missing_data_bar_plot.pdf'), format="pdf", bbox_inches='tight',
        pad_inches=0)

    # For each variable, visualize the percentage of subjects that have no
    # recorded measurement
    subject_no_values_df = subject_no_values_df.replace(False, np.nan)
    ax = missingno.bar(subject_no_values_df, color=(31/256, 119/256, 180/256))
    ax.figure.savefig(os.path.join(args.plots_path,
        'no_variable_recording_per_subject.pdf'), format="pdf",
        bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    main(parse_cl_args())

