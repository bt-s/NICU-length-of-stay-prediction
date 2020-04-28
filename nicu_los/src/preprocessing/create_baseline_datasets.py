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
from tqdm import tqdm

from ..utils.preprocessing_utils import get_first, get_last, split_train_val, \
        create_baseline_datasets_per_subject
from ..utils.utils import get_subject_dirs


def parse_cl_args():
    """Parses CL arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-sp', '--subjects-path', type=str, default='data/',
            help='Path to subject directories.')
    parser.add_argument('-pi', '--pre-imputed', type=int, default=0,
            help='Whether to use pre-imputed time-series.')
    parser.add_argument('-v', '--verbose', type=str, default=True,
            help='Console output verbosity flag.')

    return parser.parse_args(argv[1:])


def main(args):
    subjects_path = args.subjects_path
    pre_imputed = args.pre_imputed

    train_sub_path = os.path.join(subjects_path, 'training_subjects.txt')
    val_sub_path = os.path.join(subjects_path,
            'validation_subjects.txt')
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

        # Split the training directories into 80% training directories and
        # 20% validation directories
        train_dirs, val_dirs = split_train_val(os.path.join(subjects_path,
            'train/'), val_perc=0.2)

        # Write the training, validation and test subjects to files
        with open(train_sub_path,'w') as f: f.write('\n'.join(train_dirs))
        with open(val_sub_path,'w') as f: f.write('\n'.join(val_dirs))
        with open(test_sub_path,'w') as f: f.write('\n'.join(test_dirs))

    with open('nicu_los/config.json') as f:
        config = json.load(f)
        variables = config['variables']
        sub_seqs = config['baseline_subsequences']

    # Functions to compute statistical features
    stat_fns = [get_first, get_last, np.min, np.max, np.mean, np.std, skew,
            len]

    subject_dirs = train_dirs + val_dirs + test_dirs

    with mp.Pool() as pool:
        for _ in tqdm(pool.istarmap(create_baseline_datasets_per_subject,
            zip(subject_dirs, repeat(variables), repeat(stat_fns),
                repeat(sub_seqs), repeat(pre_imputed)))):
            pass


if __name__ == '__main__':
    main(parse_cl_args())

