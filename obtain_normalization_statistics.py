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


def parse_cl_args():
    """Parses CL arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-trp', '--train-path', type=str, default='data/train/',
            help='Path to the train directories.')
    parser.add_argument('-tep', '--test-path', type=str, default='data/test/',
            help='Path to the testdirectories.')
    parser.add_argument('-cp', '--config-path', type=str, default='config.json',
            help='Path to the JSON configuration file.')
    parser.add_argument('-v', '--verbose', type=int,
            help='Level of verbosity in console output.', default=1)

    return parser.parse_args(argv[1:])



def main(args):
    train_path, test_path = args.train_path, args.test_path

    train_directories = os.listdir(train_path)
    train_directories = set(filter(lambda x: str.isdigit(x), train_directories))
    train_directories = [train_path + sd for sd in train_directories]

    with open(args.config_path, 'r') as f:
        config = json.load(f)
    variables = config['variables']

    for var in variables:
        values = []
        print(f"Finding the mean and the standard deviation of {var}...")
        for subject_dir in tqdm(train_directories):
            # Read the timeseries dataframe
            df_ts = pd.read_csv(os.path.join(subject_dir,
                'timeseries_imputed.csv'))

            # Append the values of the current variable
            values = values + df_ts[var].to_list()

        config["normalization_statistics"][var]["MEAN"] = np.mean(values)
        config["normalization_statistics"][var]["STDEV"] = np.std(values)

    with open(args.config_path, 'w') as f:
        json.dump(config, f)
        f.truncate()

if __name__ == '__main__':
    main(parse_cl_args())

