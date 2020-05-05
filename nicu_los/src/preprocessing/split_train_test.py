#!/usr/bin/python3

"""split_train_test.py

Script to split the subjects into train and test sets
"""

__author__ = "Bas Straathof"

import argparse, os, random, shutil

from sys import argv


def parse_cl_args():
    """Parses CL arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-sp', '--subjects-path', type=str, default='data/',
            help='Path to subject directories.')
    parser.add_argument('-tp', '--training-percentage', type=str, default=80,
            help='Percentage of training split.')

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
        src = os.path.join(subjects_path, sd)
        dest = os.path.join(subjects_path, dir_name,  sd)
        shutil.move(src, dest)


def main(args):
    random.seed(21128947124)
    subjects_path = args.subjects_path
    perc = args.training_percentage / 100

    subject_dirs = os.listdir(subjects_path)
    subject_dirs = list(set(filter(lambda x: str.isdigit(x),
        subject_dirs)))
    tot_dirs = len(subject_dirs)
    split = round(tot_dirs*perc)

    random.shuffle(subject_dirs)

    train_dirs = subject_dirs[:split]
    test_dirs = subject_dirs[split:]

    print(f'There are {len(train_dirs)} train directories ' \
            f'and {len(test_dirs)} test directories.')

    # Create train and test directories
    move_to_directory(subjects_path, train_dirs, 'train')
    move_to_directory(subjects_path, test_dirs, 'test')

if __name__ == '__main__':
    main(parse_cl_args())

