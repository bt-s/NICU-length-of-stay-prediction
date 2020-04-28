#!/usr/bin/python3

"""obtain_normalization_statistics.py

Script to obtain statistics for timeseries normalization
"""

__author__ = "Bas Straathof"


import argparse, json, os

from sys import argv
from tqdm import tqdm

import pandas as pd
import numpy as np

from ..utils.utils import get_subject_dirs


def parse_cl_args():
    """Parses CL arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-trp', '--train-path', type=str, default='data/train/',
            help='Path to the train directories.')
    parser.add_argument('-v', '--verbose', type=int,
            help='Level of verbosity in console output.', default=1)

    return parser.parse_args(argv[1:])


def main(args):
    train_path = args.train_path

    train_dirs = get_subject_dirs(train_path)

    with open('nicu_los/config.json', 'r') as f:
        config = json.load(f)
    variables = config['variables']

    for var in variables:
        values = []
        print(f"Finding the mean and the standard deviation of {var}...")
        for subject_dir in tqdm(train_dirs):
            # Read the timeseries dataframe
            df_ts = pd.read_csv(os.path.join(subject_dir,
                'timeseries_imputed.csv'))

            # Append the values of the current variable
            values = values + df_ts[var].to_list()

        config["normalization_statistics"][var]["MEAN"] = np.mean(values)
        config["normalization_statistics"][var]["STDEV"] = np.std(values)

    with open('nicu_los/config.json', 'w') as f:
        json.dump(config, f)
        f.truncate()

if __name__ == '__main__':
    main(parse_cl_args())

