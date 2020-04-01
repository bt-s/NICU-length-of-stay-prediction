#!/usr/bin/python3

"""create_timeseries.py

Script to create timeseries from preprocessed the clinical events tables
"""

__author__ = "Bas Straathof"

import argparse, json, os

import pandas as pd
import numpy as np

from tqdm import tqdm
from sys import argv

from utils import round_up_to_hour, compute_ga_days_for_charttime, \
        compute_remaining_los, los_hours_to_target, \
        get_first_valid_value_from_ts, remove_subject_dir


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


def create_timeseries(variables, df_events, df_stay, df_notes=None):
    """Create timeseries from clinical events (and notes)

    Args:
        variables (dict): All variables to be present in the timeseries
        df_events: Dataframe containing all clinical events of ICU stay
        df_stay: Dataframe containing information about ICU stay
        df_notes: Dataframe containing notes about ICU stay

    Returns:
        df_ts: Dataframe containing the timesries of the ICU stay
    """
    # If not hour on clock, round up charttime to nearest hour
    df_events.CHARTTIME = df_events.CHARTTIME.apply(
            lambda x: round_up_to_hour(x))

    # Round up intime
    intime = round_up_to_hour(df_stay.iloc[0]['INTIME'])

    # Sort df_events by CHARTTIME; only keep the last value of a variable per
    # timestamp
    df_ts = df_events[['CHARTTIME', 'VARIABLE', 'VALUE', 'ITEMID']] \
            .sort_values(by=['CHARTTIME', 'VARIABLE', 'VALUE'], axis=0) \
            .drop_duplicates(subset=['CHARTTIME', 'VARIABLE'], keep='last')

    # Only keep first birth weight
    df_ts = df_ts[~(df_ts.duplicated(['ITEMID'], keep='first') &
        df_ts.ITEMID.isin([3723, 4183]))]

    # Pivot the dataframe s.t. the column names are the variables
    df_ts = df_ts.pivot(index='CHARTTIME', columns='VARIABLE',
            values='VALUE').sort_index(axis=0).reset_index()

    # Make sure that the timeeries contains all variables
    for v in variables:
        if v not in df_ts:
            df_ts[v] = np.nan

    # Make sure that if WEIGHT and HEIGHT values are present, that the first
    # first row in the timeseries contain a value
    df_ts.WEIGHT.iloc[0] = get_first_valid_value_from_ts(df_ts, 'WEIGHT')
    df_ts.HEIGHT.iloc[0] = get_first_valid_value_from_ts(df_ts, 'HEIGHT')

    # Add GA days to timeseries
    ga_days = df_stay.iloc[0].GA_DAYS
    df_ts.GESTATIONAL_AGE_DAYS = df_ts['CHARTTIME'].apply(lambda x:
            compute_ga_days_for_charttime(x, intime, ga_days))

    # One-hot-encode the capillary refill rate variable
    df_ts = pd.get_dummies(df_ts, prefix=['CAPILLARY_REFILL_RATE'],
            columns=['CAPILLARY_REFILL_RATE'])

    # Add target LOS to timeseries
    los_hours = df_stay.iloc[0].LOS_HOURS
    df_ts['TARGET'] = df_ts['CHARTTIME'].apply(lambda x:
            los_hours_to_target(compute_remaining_los(x, intime, los_hours)))

    return df_ts


def main(args):
    verbose, subjects_path = args.verbose, args.subjects_path
    removed_subjects, tot_events, tot_events_kept = 0, 0, 0

    with open(args.config_path) as f:
        config = json.load(f)
        variables = config['variables']

    subject_directories = os.listdir(subjects_path)
    subject_directories = set(filter(lambda x: str.isdigit(x),
        subject_directories))
    tot_subjects = len(subject_directories)

    for i, subject_dir in enumerate(tqdm(subject_directories)):
        # Read the events dataframe
        df_events = pd.read_csv(os.path.join(subjects_path, str(subject_dir),
            'events.csv'))

        # Read the admission dataframe
        df_stay = pd.read_csv(os.path.join(subjects_path, str(subject_dir),
            'stay.csv'))

        # Create the timeseries
        df_ts = create_timeseries(variables, df_events, df_stay)

        # Write timeseries to timeseries.csv if not empty, remove otherwise
        if not df_ts.empty:
            df_ts.to_csv(os.path.join(subjects_path, str(subject_dir),
                'timeseries.csv'), index=False)
        else:
            remove_subject_dir(os.path.join(subjects_path, str(subject_dir)))
            removed_subjects += 1

    if verbose:
        print(f'Of the initial {tot_subjects} subjects, ' \
                f'{tot_subjects-removed_subjects} remain that have ' \
                f'non-empty time-series.\n')

if __name__ == '__main__':
    main(parse_cl_args())

