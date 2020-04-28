#!/usr/bin/python3

"""normalize_values.py

Script to normalize values in timeseries
"""

__author__ = "Bas Straathof"

import argparse, json, os

from tqdm import tqdm
from sys import argv
from itertools import repeat

import multiprocessing as mp
import pandas as pd
import numpy as np

from ..utils.utils import get_subject_dirs


def parse_cl_args():
    """Parses CL arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-trp', '--train-path', type=str, default='data/train/',
            help='Path to the train directories.')
    parser.add_argument('-tep', '--test-path', type=str, default='data/test/',
            help='Path to the testdirectories.')
    parser.add_argument('-v', '--verbose', type=int,
            help='Level of verbosity in console output.', default=1)

    return parser.parse_args(argv[1:])


def normalize(sd, normalization_statistics, variables):
    """Normalize a timeseries corresponding to a subject

    Args:
        sd (str): Path to the subject directory
        normalization_statistics (dict): Dictionary containing the means and
                                         standard deviations of the variables
                                         of interest
        variables (list): List of variables of interest
    """
    # Read the timeseries dataframe
    ts = pd.read_csv(os.path.join(sd, 'timeseries_imputed.csv'))
    ts = ts.set_index('CHARTTIME')

    for var in variables:
        mean = normalization_statistics[var]["MEAN"]
        stdev = normalization_statistics[var]["STDEV"]

        ts[var] = ts[var].apply(lambda x: (x - mean) / stdev)

    # Write the timeseries to CSV
    ts.to_csv(os.path.join(sd, 'timeseries_normalized.csv'))


def main(args):
    with open('nicu_los/config.json') as f:
        config = json.load(f)
        normalization_statistics = config['normalization_statistics']
        variables = config['variables']

    train_dirs = get_subject_dirs(args.train_path)
    test_dirs = get_subject_dirs(args.test_path)
    all_dirs = train_dirs + test_dirs

    with mp.Pool() as pool:
        for _ in tqdm(pool.istarmap(normalize, zip(all_dirs,
            repeat(normalization_statistics), repeat(variables))),
            total=len(all_dirs)):
            pass


if __name__ == '__main__':
    main(parse_cl_args())

