#!/usr/bin/python3

"""createtime_series.py

Script to create timeseries from preprocessed the clinical events tables
"""

__author__ = "Bas Straathof"

import argparse, json, os

import pandas as pd
import numpy as np

from tqdm import tqdm
from sys import argv

from nicu_los.src.utils.utils import compute_ga_days_for_charttime, \
        compute_remaining_los, get_subject_dirs, remove_subject_dir, \
        round_up_to_hour


def parse_cl_args():
    """Parses CL arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-sp', '--subjects-path', type=str, default='data/',
            help='Path to subject directories.')

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

    # Add target LOS to timeseries
    los_hours = df_stay.iloc[0].LOS_HOURS
    df_ts['LOS_HOURS'] = df_ts['CHARTTIME'].apply(lambda x: \
            compute_remaining_los(x, intime, los_hours))
    df_ts['TARGET_FINE'] = df_ts['LOS_HOURS'].apply(lambda x: \
            los_hours_to_target(x, coarse=False))
    df_ts['TARGET_COARSE'] = df_ts['LOS_HOURS'].apply(lambda x: \
            los_hours_to_target(x, coarse=True))

    return df_ts


def get_first_valid_value_from_ts(ts, variable):
    """Get the first valid value of a variable in a time series

    Args:
        ts (pd.DataFrame): Timeseries dataframe
        variable (str): Name of the variable

    Returns:
        value (float): First valid value of variable
    """
    # Assume no value exists
    value = np.nan
    if variable in ts:
        # Find the indices of the rows where variable has a value in ts
        indices = ts[variable].notnull()
        if indices.any():
            index = indices.to_list().index(True)
            value = ts[variable].iloc[index]

    return value


def los_hours_to_target(hours, coarse):
    """Convert LOS in hours to targets

    The fine targets exist of ten buckets:
        0: less than 1 day
        1: 1-2 days
        2: 2-3 days
        3: 3-4 days
        4: 4-5 days
        5: 5-6 days
        6: 6-7 days
        7: 7-8 days
        8: 8-13 days
        9: more than 14 days

    The coarse targets exist of three buckets:
        0: less than 3 days;
        1: 3-7 days;
        2: more than 7 days

    Args:
        hours (int): LOS in hours
        coarse (bool): Whether to use coarse labelling

    Return:
        target (int): The respective target
    """
    if not coarse:
        if hours < 24:
            target = 0
        elif 24 <= hours < 48:
            target = 1
        elif 48 <= hours < 72:
            target = 2
        elif 72 <= hours < 96:
            target = 3
        elif 96 <= hours < 120:
            target = 4
        elif 120 <= hours < 144:
            target = 5
        elif 144 <= hours < 168:
            target = 6
        elif 168 <= hours < 192:
            target = 7
        elif 192 <= hours < 336:
            target = 8
        elif 336 <= hours:
            target = 9
    else:
        if hours < 48:
            target = 0
        elif 48 <= hours < 168:
            target = 1
        elif 168 <= hours:
            target = 2

    return target


def main(args):
    subjects_path = args.subjects_path
    removed_subjects, tot_events, tot_events_kept = 0, 0, 0

    with open('nicu_los/config.json') as f:
        config = json.load(f)
        variables = config['variables']

    subject_dirs = get_subject_dirs(subjects_path)
    tot_subjects = len(subject_dirs)

    for i, sd in enumerate(tqdm(subject_dirs)):
        # Read the events dataframe
        df_events = pd.read_csv(os.path.join(sd, 'events.csv'))

        # Read the admission dataframe
        df_stay = pd.read_csv(os.path.join(sd, 'stay.csv'))

        # Create the timeseries
        df_ts = create_timeseries(variables, df_events, df_stay)

        # Write timeseries to timeseries.csv if not empty, remove otherwise
        if not df_ts.empty:
            df_ts.to_csv(os.path.join(sd, 'timeseries.csv'), index=False)
        else:
            remove_subject_dir(os.path.join(sd))
            removed_subjects += 1

    print(f'Of the initial {tot_subjects} subjects, ' \
            f'{tot_subjects-removed_subjects} remain that have ' \
            f'non-empty time-series.\n')

if __name__ == '__main__':
    main(parse_cl_args())

