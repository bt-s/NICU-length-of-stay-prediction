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
    parser.add_argument('-v', '--verbose', type=int,
            help='Level of verbosity in console output.', default=1)

    return parser.parse_args(argv[1:])


def move_to_directory(subjects_path, subject_directories, dir_name):
    """Move subjects to train/test directory

    Args:
        subjects_path (str): Path to the subject directories
        subject_directories (list): List of subject directories
        dir_name (str): Name of split directory (train/test)
    """
    path = os.path.join(subjects_path, dir_name)
    if not os.path.exists(path):
        os.makedirs(path)

    for subject_dir in subject_directories:
        src = os.path.join(subjects_path, subject_dir)
        dest = os.path.join(subjects_path, dir_name,  subject_dir)
        shutil.move(src, dest)


def main(args):
    random.seed(21128947124)
    verbose, subjects_path = args.verbose, args.subjects_path
    perc = args.training_percentage / 100

    subject_directories = os.listdir(subjects_path)
    subject_directories = list(set(filter(lambda x: str.isdigit(x),
        subject_directories)))
    tot_directories = len(subject_directories)
    split = round(tot_directories*perc)

    random.shuffle(subject_directories)

    train_directories = subject_directories[:split]
    test_directories = subject_directories[split:]

    # Create train and test directories
    move_to_directory(subjects_path, train_directories, 'train')
    move_to_directory(subjects_path, test_directories, 'test')

if __name__ == '__main__':
    main(parse_cl_args())

