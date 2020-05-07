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
import multiprocessing as mp
from itertools import repeat

from nicu_los.src.utils.utils import get_subject_dirs


def parse_cl_args():
    """Parses CL arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-trp', '--train-path', type=str, default='data/train',
            help='Path to the train directories.')

    return parser.parse_args(argv[1:])


def get_normalization_stats_for_var(variable, train_dirs, config, q):
    """Worker function

    Args:
        variable (str): Variable for which to obtain the normalization statistics
        train_dirs (list): List of training directories
        config (dict): Configuration file
        q (mp.Manager.Queue): Multiprocessing queue manager
    
    Sends the mean and standard deviation of variable to the queue mananger
    """
    values = []
    for subject_dir in tqdm(train_dirs):
        # Read the timeseries dataframe
        df_ts = pd.read_csv(os.path.join(subject_dir,
            'timeseries_imputed.csv'))

        # Append the values of the current variable
        values = values + df_ts[variable].tolist()

    print(f'{variable} MEAN: {np.mean(values)}')
    print(f'{variable} STDEV: {np.std(values)}')

    q.put((variable, np.mean(values), np.std(values)))


def listener(config, q):
    """Listener function to ensure safe writing to config file
    
    Args:
        config (dict): Configuration file
        q (mp.Manager.Queue): Multiprocessing queue manager
        
    Writes the updated config to file
    """
    with open('nicu_los/config.json', 'w') as f:
        while True:
            m = q.get()

            if m == 'kill':
                json.dump(config, f)
                f.truncate()
                break
            else:
                config["normalization_statistics"][m[0]]["MEAN"] = m[1]
                config["normalization_statistics"][m[0]]["STDEV"] = m[2]


def main(args):
    train_dirs = get_subject_dirs(args.train_path)

    with open('nicu_los/config.json', 'r') as f:
        config = json.load(f)
        variables = config['variables']

    manager = mp.Manager()
    q = manager.Queue()
    pool = mp.Pool()

    # Create a listener s.t. it is safe to write to the config file
    watcher = pool.apply_async(listener, (config, q,))

    # Create worker processes
    jobs = []
    for variable in variables:
        job = pool.apply_async(get_normalization_stats_for_var,
                (variable, train_dirs, config, q))
        jobs.append(job)

    # Collect results rom the pool result queue
    for job in jobs:
        job.get()

    # Kill the listener once all jobs are done
    q.put('kill')
    pool.close()
    pool.join()


if __name__ == '__main__':
    main(parse_cl_args())

