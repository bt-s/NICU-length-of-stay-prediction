#!/usr/bin/python3

"""target_distributions.py

Script to visualize the distribution of the targets over the data set.
"""

__author__ = "Bas Straathof"

import argparse, errno, os

import pandas as pd
import numpy as np

from sys import argv
from tqdm import tqdm

from nicu_los.src.utils.utils import get_subject_dirs
from nicu_los.src.utils.visualization import create_histogram


def parse_cl_args():
    """Parses CL arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-sp', '--subjects-path', type=str, default='data',
            help='Path to subject directories.')
    parser.add_argument('-pp', '--plots-path', type=str, default='plots/',
            help='Path to plots directory.')

    return parser.parse_args(argv[1:])


def main(args):
    train_path = os.path.join(args.subjects_path, 'train')
    test_path = os.path.join(args.subjects_path, 'test')

    if not (os.path.exists(train_path) or os.path.exists(test_path)):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), train_path)
    elif not (os.path.exists(args.plots_path)): os.makedirs(args.plots_path)

    subject_directories_train = get_subject_dirs(train_path)
    subject_directories_test = get_subject_dirs(test_path)
    subject_directories = subject_directories_train + subject_directories_test

    los_hours, los_remaining_hours, los_targets_coarse, \
            los_remaining_targets_coarse, los_targets_fine, \
            los_remaining_targets_fine =  [], [], [], [], [], []

    for i, sd in enumerate(tqdm(subject_directories)):
        # Read the timeseries dataframe
        ts = pd.read_csv(os.path.join(sd, 'timeseries.csv'))

        # Find the total length of the stay in hours
        los_hours.append(ts.LOS_HOURS.iloc[0])

        # Compute the coarse target bucket for the complete stay
        los_targets_coarse.append(ts.TARGET_COARSE.iloc[0])

        # Compute the fine target bucket for the complete stay
        los_targets_fine.append(ts.TARGET_FINE.iloc[0])

        # Find all the intermediate remaining length of stay in hours
        los_remaining_hours += ts.LOS_HOURS.to_list()

        # Obtain the coarse target bucket for each intermediate time-step
        los_remaining_targets_coarse += ts.TARGET_COARSE.to_list()

        # Obtain the fine target bucket for each intermediate time-step
        los_remaining_targets_fine += ts.TARGET_FINE.to_list()

    # Only keep the 95% percentile of los_hours and los_remaining_hours
    los_perc = np.percentile(los_hours, 95)
    los_remaining_perc = np.percentile(los_remaining_hours, 95)

    los_hours = list(filter(lambda x: x < los_perc, los_hours))
    los_remaining_hours = list(filter(lambda x: x < los_remaining_perc,
            los_remaining_hours))

    # X-ticks for the coarse buckets plot
    xticks_coarse = ['(0, 2)', '(2, 8)', '8+']

    # X-ticks for the fine buckets plot
    xticks_fine = ['(0, 1)', '(1, 2)', '(2, 3)', '(3, 4)', '(4, 5)', '(5, 6)',
            '(6, 7)', '(7, 8)', '(8, 14)', '14+']

    # Create the coarse buckets histogram
    create_histogram(input_data=[los_targets_coarse,
        los_remaining_targets_coarse], xlabel='Buckets', ylabel='Frequency',
        rwidth=0.5, legend=['LOS', 'Remaining LOS'], xticks=xticks_coarse,
        save_plot=(os.path.join(args.plots_path,
            'normalized_frequency_of_the_target_buckets_coarse.pdf')))

    # Create the fine buckets histogram
    create_histogram(input_data=[los_targets_fine, los_remaining_targets_fine], 
            xlabel='Buckets', ylabel='Frequency', rwidth=0.5,
            legend=['LOS', 'Remaining LOS'], xticks=xticks_fine, 
            save_plot=(os.path.join(args.plots_path,
                'normalized_frequency_of_the_target_buckets_fine.pdf')))

    # Create the LOS hours histogram
    create_histogram(input_data=[los_hours, los_remaining_hours],
            xlabel='Hours', ylabel='Frequency', rwidth=1,
            legend=['LOS', 'Remaining LOS'], save_plot=(os.path.join(
                args.plots_path, 'normalized_frequency_of_the_LOS_in_hours.pdf')))


if __name__ == '__main__':
    main(parse_cl_args())
