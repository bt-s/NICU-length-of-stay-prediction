#!/usr/bin/python3

"""createtime_series.py

Script to create timeseries from preprocessed the clinical events tables
"""

__author__ = "Bas Straathof"

import argparse, json, os

import pandas as pd
import numpy as np
import multiprocessing as mp

from itertools import repeat
from tqdm import tqdm
from sys import argv

from nicu_los.src.utils.utils import compute_ga_days_for_charttime, \
        compute_remaining_los, get_subject_dirs, istarmap, \
        remove_subject_dir, round_up_to_hour


def parse_cl_args():
    """Parses CL arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-sp', '--subjects-path', type=str, default='data',
            help='Path to subject directories.')

    parser.add_argument('-c', '--config', type=str,
            default='nicu_los/config.json', help='Path to the config file')

    return parser.parse_args(argv[1:])


def create_timeseries(variables, subject_dir):
    """Create timeseries from clinical events (and notes)

    Args:
        variables (dict): All variables to be present in the timeseries
        subject_dir (list): List of subject directories

    Returns:
        df_ts: Dataframe containing the timesries of the ICU stay
    """
    # Read the events dataframe
    df_events = pd.read_csv(os.path.join(subject_dir, 'events.csv'))

    # Read the admission dataframe
    df_stay = pd.read_csv(os.path.join(subject_dir, 'stay.csv'))

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

    # Make sure that the timeseries contains all variables
    for v in variables:
        if v not in df_ts:
            df_ts[v] = np.nan

    # Make sure that if WEIGHT and HEIGHT values are present, that the first
    # first row in the timeseries contain a value
    df_ts.WEIGHT.iloc[0] = get_first_valid_value_from_ts(df_ts, 'WEIGHT')
    df_ts.HEIGHT.iloc[0] = get_first_valid_value_from_ts(df_ts, 'HEIGHT')

    # Make sure that there is a time stamp for each hour
    df_ts = df_ts.set_index('CHARTTIME')
    df_ts = df_ts.reindex(pd.date_range(start=df_ts.index[0],
        end=df_ts.index[-1], freq='3600S'))
    df_ts['CHARTTIME'] = df_ts.index
    df_ts.index = pd.RangeIndex(len(df_ts.index))

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

    if not df_ts.empty:
        df_ts.to_csv(os.path.join(subject_dir, 'timeseries.csv'), index=False)
    else:
        remove_subject_dir(os.path.join(subject_dir))


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

    with open(args.config) as f:
        config = json.load(f)
        variables = config['variables']

    subject_dirs = get_subject_dirs(subjects_path)
    tot_subjects = len(subject_dirs)

    with mp.Pool() as pool:
        for _ in tqdm(pool.istarmap(create_timeseries, zip(repeat(variables),
            subject_dirs)), total=tot_subjects):
            pass


if __name__ == '__main__':
    main(parse_cl_args())

