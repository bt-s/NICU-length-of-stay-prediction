#!/usr/bin/python3

"""impute_values.py

Script to impute missing values in timeseries
"""

__author__ = "Bas Straathof"

import argparse, json, math, os

from tqdm import tqdm
from sys import argv
from itertools import repeat

import multiprocessing as mp
import pandas as pd

from utils import istarmap


def parse_cl_args():
    """Parses CL arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-sp', '--subjects-path', type=str, default='data/',
            help='Path to subject directories.')
    parser.add_argument('-cp', '--config-path', type=str, default='config.json',
            help='Path to the JSON configuration file.')
    parser.add_argument('-v', '--verbose', type=int,
            help='Level of verbosity in console output.', default=1)

    return parser.parse_args(argv[1:])


def impute(subject_dir, normal_values, mask=True):
    ts = pd.read_csv(os.path.join(subject_dir, 'timeseries.csv'))
    ts = ts.set_index('CHARTTIME')

    variables = list(normal_values.keys())

    if mask:
        # Create an imputation mask
        ts_mask = ts.mask(ts.notna(), 1)
        ts_mask = ts_mask.mask(ts.isna(), 0)
        ts_mask = ts_mask.drop(['TARGET'], axis=1)
        ts_mask = ts_mask.add_prefix('mask_')

    # Make sure that the first row contains values such that we can
    # do a forward fill impute
    for var in variables:
        if math.isnan(ts[var].iloc[0]):
            if var == 'WEIGHT' or var == 'HEIGHT':
                ts[var].iloc[0] = normal_values[var] \
                        [str(int(round(ts['GESTATIONAL_AGE_DAYS'].iloc[0] / 7)
                            ))]
            else:
                ts[var].iloc[0] = normal_values[var]

    # Impute through forward filling
    ts = ts.fillna(method='ffill')

    if mask:
        # Concatenate the timeseries with the imputation mask
        ts = pd.concat([ts, ts_mask], axis=1)

    ts.to_csv(os.path.join(subject_dir, 'timeseries_imputed.csv'))


def main(args):
    verbose, subjects_path = args.verbose, args.subjects_path

    with open(args.config_path) as f:
        config = json.load(f)
        normal_values = config['normal_values']

    subject_directories = os.listdir(subjects_path)
    subject_directories = set(filter(lambda x: str.isdigit(x),
        subject_directories))
    subject_directories = [subjects_path + sd for sd in subject_directories]

    with mp.Pool() as pool:
        for _ in tqdm(pool.istarmap(impute, zip(subject_directories,
            repeat(normal_values))), total=len(subject_directories)):
            pass


if __name__ == '__main__':
    main(parse_cl_args())

