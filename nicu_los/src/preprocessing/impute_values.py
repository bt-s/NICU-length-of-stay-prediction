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

from nicu_los.src.utils.utils import get_subject_dirs, istarmap


def parse_cl_args():
    """Parses CL arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-sp', '--subjects-path', type=str, default='data',
            help='Path to subject directories.')
    parser.add_argument('-im', '--impute-method', type=str, default='ffill',
            help='Impute method to use: either "ffill" or "zeros".')
    parser.add_argument('-ma', '--mask', type=str, default=True,
            help='Whether to create binary imputation indicator variables.')
    parser.add_argument('-c', '--config', type=str,
            default='nicu_los/config.json', help='Path to the config file')

    return parser.parse_args(argv[1:])


def impute(sd, imputation_values, method='ffill', mask=True):
    """Forward fill impute missing values with normal values

    Args:
        sd (str): String to the subject directory
        imputation_values (dict): Normal values for selected variables
        method (str): Either 'ffill' or 'zeros' -- determines impute method
        mask (bool): Whether to create binary imputation masks
    """
    ts = pd.read_csv(os.path.join(sd, 'timeseries.csv'))
    ts = ts.set_index('CHARTTIME')

    variables = list(imputation_values.keys())

    if mask:
        # Create an imputation mask
        ts_mask = ts.mask(ts.notna(), 1)
        ts_mask = ts_mask.mask(ts.isna(), 0)
        ts_mask = ts_mask.drop(['LOS_HOURS', 'TARGET_COARSE', 'TARGET_FINE'],
                axis=1)
        ts_mask = ts_mask.add_prefix('mask_')

    # Make sure that the first row contains values such that we can
    # do a forward fill impute
    for var in variables:
        if math.isnan(ts[var].iloc[0]):
            if var == 'WEIGHT' or var == 'HEIGHT':
                ts[var].iloc[0] = imputation_values[var] \
                        [str(int(round(ts['GESTATIONAL_AGE_DAYS'].iloc[0] / 7)
                            ))]
            else:
                ts[var].iloc[0] = imputation_values[var]


    if method == 'ffill':
        # Impute through forward filling
        ts = ts.fillna(method='ffill')
    elif method == 'zeros':
        # Impute through filling with zeros
        ts = ts.fillna(value=0)
    else:
        raise ValueError(f'{method} must be one of "ffill" or "zeros"')

    if mask:
        # Concatenate the timeseries with the imputation mask
        ts = pd.concat([ts, ts_mask], axis=1)

    ts.to_csv(os.path.join(sd, 'timeseries_imputed.csv'))


def main(args):
    impute_method = args.impute_method

    with open(args.config) as f:
        config = json.load(f)
        imputation_values = config['imputation_values']

    print(f'Starting {impute_method} imputing with normal values.' \
            f'Binary imputation mask: {args.mask}')

    subject_dirs = get_subject_dirs(args.subjects_path)

    with mp.Pool() as pool:
        for _ in tqdm(pool.istarmap(impute, zip(subject_dirs,
            repeat(imputation_values), repeat(impute_method),
            repeat(args.mask))), total=len(subject_dirs)):
            pass


if __name__ == '__main__':
    main(parse_cl_args())

