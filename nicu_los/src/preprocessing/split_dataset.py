#!/usr/bin/python3

"""split_dataset.py

Script to split the subjects into train, validation and test sets
"""

__author__ = "Bas Straathof"

import argparse, os, random, shutil

from sys import argv
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

from nicu_los.src.utils.utils import get_subject_dirs


def parse_cl_args():
    """Parses CL arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-sp', '--subjects-path', type=str, default='data',
            help='Path to subject directories.')
    parser.add_argument('-tp', '--train-perc', type=str, default=80,
            help='Percentage of training split.')
    parser.add_argument('-vp', '--val-perc', type=str, default=80,
            help='Percentage of validation split.')

    return parser.parse_args(argv[1:])


def move_to_directory(subjects_path, subject_dirs, dir_name):
    """Move subjects to train/test directory

    Args:
        subjects_path (str): Path to the subject directories
        subject_dirs (list): List of subject directories
        dir_name (str): Name of split directory (train/test)
    """
    path = os.path.join(subjects_path, dir_name)
    if not os.path.exists(path):
        os.makedirs(path)

    for sd in subject_dirs:
        subject_id = ''.join(x for x in sd if x.isdigit())
        dest = os.path.join(subjects_path, dir_name,  subject_id)
        shutil.move(sd, dest)


def split_data_set(data_dirs_path, split_perc=20, bin_size=4):
    """Split the data set into two (x and y)

    Args:
        data_dirs_path (str): Path to the data directories
        val_perc (int): Percentage of data to be reserved for validation
        bin_size (int): Minimum amount of values per bin

    Returns:
        x_dirs (list): List of x-split directories
        y_dirs (list): List of y-split directories
    """
    data_dirs = get_subject_dirs(data_dirs_path)

    # Get two arrays: one of targets and one of the
    # corresponding subjects
    targets = np.zeros(len(data_dirs))
    subjects = np.zeros(len(data_dirs))
    for i, sd in enumerate(data_dirs):
        df_ts = pd.read_csv(os.path.join(sd, 'timeseries.csv'))
        targets[i] = df_ts.LOS_HOURS.iloc[0]
        subject_id = [int(s) for s in sd.split('/') if s.isdigit()][-1]
        subjects[i] = subject_id

    # Define the bins for splitting
    sorted_targets = np.sort(targets)
    bins = [0]
    set_check = set()
    for t in np.sort(targets):
        set_check.add(t)
        if len(set_check) > bin_size:
            bins.append(t)
            set_check = set()
    bins.append(max(targets)+1)

    # Bin the targets
    targets_binned = np.digitize(targets, bins)

    # Split the subjects list into a list of x-subjects and a
    # list of y-subjects, in a stratified manner
    subjects_y, subjects_x, _, _ = train_test_split(
            subjects, targets, test_size=split_perc/100, random_state=42,
            stratify=targets_binned, shuffle=True)

    x_dirs = [f'{data_dirs_path}/{int(i)}' for i in subjects_x]
    y_dirs = [f'{data_dirs_path}/{int(i)}' for i in subjects_y]

    return x_dirs, y_dirs


def main(args):
    subjects_path = args.subjects_path

    # Split the data set into training and test data 
    train_dirs, test_dirs = split_data_set(subjects_path, args.train_perc)

    print(f'There are {len(train_dirs)} train directories ' \
            f'and {len(test_dirs)} test directories.')

    # Create train and test directories
    move_to_directory(subjects_path, train_dirs, 'train')
    move_to_directory(subjects_path, test_dirs, 'test')

    print('...split the training set into training and validation...')
    train_dirs, val_dirs = split_data_set(os.path.join(subjects_path,
        'train'), args.val_perc, bin_size=9) # larger bin size because less data
    test_dirs = get_subject_dirs(os.path.join(subjects_path, 'test'))

    print(f'There are {len(train_dirs)} train directories ' \
            f'and {len(val_dirs)} validation directories.')

    train_sub_path = os.path.join(subjects_path, 'training_subjects.txt')
    val_sub_path = os.path.join(subjects_path, 'validation_subjects.txt')
    test_sub_path = os.path.join(subjects_path, 'test_subjects.txt')

    print('...write the training, validation and test subjects to files...')
    with open(train_sub_path,'w') as f: f.write('\n'.join(train_dirs))
    with open(val_sub_path,'w') as f: f.write('\n'.join(val_dirs))
    with open(test_sub_path,'w') as f: f.write('\n'.join(test_dirs))


if __name__ == '__main__':
    main(parse_cl_args())

