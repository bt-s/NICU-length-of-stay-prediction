#!/usr/bin/python3

"""target_distributions.py

Script to visualize the distribution of the targets over the data set.
"""

__author__ = "Bas Straathof"

import argparse, errno, os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sys import argv
from tqdm import tqdm

from ..utils.utils import compute_remaining_los, get_subject_dirs, \
        round_up_to_hour
from ..utils.preprocessing_utils import los_hours_to_target
from ..utils.visualization_utils import create_histogram


def parse_cl_args():
    """Parses CL arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-sp', '--subjects-path', type=str, default='data/',
            help='Path to subject directories.')
    parser.add_argument('-pp', '--plots-path', type=str, default='plots/',
            help='Path to plots directory.')
    parser.add_argument('-v', '--verbose', type=int,
            help='Level of verbosity in console output.', default=1)

    return parser.parse_args(argv[1:])


def main(args):
    train_path = os.path.join(args.subjects_path, 'train/')
    test_path = os.path.join(args.subjects_path, 'test/')

    if not (os.path.exists(train_path) or os.path.exists(test_path)):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), train_path)
    elif not (os.path.exists(args.plots_path)): os.makedirs(args.plots_path)

    subject_directories_train = get_subject_dirs(train_path)
    subject_directories_test = get_subject_dirs(test_path)
    subject_directories = subject_directories_train + subject_directories_test

    los_hours, los_targets = [], []
    los_remaining_hours, los_remaining_targets = [], []

    for i, sd in enumerate(tqdm(subject_directories)):
        # Read the stay dataframe and obtain the intime and total hour of stay
        stay = pd.read_csv(os.path.join(sd, 'stay.csv'))
        intime = round_up_to_hour(stay.iloc[0].INTIME)
        los_hours_tot = stay.iloc[0].LOS_HOURS

        # Read the timeseries dataframe
        ts = pd.read_csv(os.path.join(sd, 'timeseries.csv'))
        ts.CHARTTIME = ts.CHARTTIME.apply(lambda x: round_up_to_hour(x))

        # Compute the reamining LOS in hours for each charrtime
        ts['LOS_HOURS_REMAINING'] = ts.CHARTTIME.apply(lambda x:
                compute_remaining_los(x, intime, los_hours_tot))

        # Find the total length of the stay in hours
        tot_hours = stay.LOS_HOURS.iloc[0]
        los_hours.append(tot_hours)

        # Compute the target bucket for the complete stay
        los_targets.append(los_hours_to_target(tot_hours))

        # Find all the intermediate remaining length of stay in hours
        tot_hours_remaining = ts.LOS_HOURS_REMAINING.to_list()
        los_remaining_hours += tot_hours_remaining

        # Obtain the target bucket for each intermediate time-step
        los_remaining_targets += ts.TARGET.to_list()

    # Only keep the 95% percentile of los_hours and los_remaining_hours
    los_perc = np.percentile(los_hours, 95)
    los_remaining_perc = np.percentile(los_remaining_hours, 95)

    los_hours = list(filter(lambda x: x < los_perc, los_hours))
    los_remaining_hours = list(filter(lambda x: x < los_remaining_perc,
            los_remaining_hours))

    # X-ticks for the buckets plot
    xticks = ['(0, 1)', '(1, 2)', '(2, 3)', '(3, 4)', '(4, 5)', '(5, 6)',
            '(6, 7)', '(7, 8)', '(8, 14)', '14+']

    # Create the buckets histogram
    create_histogram(input_data=[los_targets, los_remaining_targets],
            xlabel='Buckets', ylabel='Frequency', rwidth=0.5,
            legend=['LOS', 'Remaining LOS'], xticks=xticks,
            save_plot=(os.path.join(args.plots_path,
                'normalized_frequency_of_the_target_buckets')))

                # Create the LOS hours histogram
    create_histogram(input_data=[los_hours, los_remaining_hours],
            xlabel='Hours', ylabel='Frequency', rwidth=1,
            legend=['LOS', 'Remaining LOS'], save_plot=(os.path.join(
                args.plots_path, 'normalized_frequency_of_the_LOS_in_hours')))


if __name__ == '__main__':
    main(parse_cl_args())
