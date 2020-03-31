import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from sys import argv
from utils import round_up_to_hour, compute_ga_weeks_for_charttime, \
        compute_remaining_los, los_hours_to_target, get_first_valid_value


def parse_cl_args():
    """Parses CL arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-sp', '--subjects-path', type=str, default='data/',
            help='Path to subject directories.')
    parser.add_argument('-v', '--verbose', type=int,
            help='Level of verbosity in console output.', default=1)

    return parser.parse_args(argv[1:])

variables = [
    'Bilirubin -- Direct',
    'Bilirubin -- Indirect',
    'Blood Pressure -- Diastolic',
    'Blood Pressure -- Systolic',
    'Capillary Refill Rate',
    'Fraction Inspired Oxygen',
    'Gestational Age -- Weeks',
    'Heart Rate',
    'Height',
    'Oxygen Saturation',
    'pH',
    'Respiratory Rate',
    'TARGET',
    'Temperature',
    'Weight'
]

def create_timeseries(df_events, df_admit, variables=variables):
    # If not hour on clock, round up CHARTTIME to nearest hour
    df_events.CHARTTIME = df_events.CHARTTIME.apply(
            lambda x: round_up_to_hour(x))

    intime = round_up_to_hour(df_admit.iloc[0]['INTIME'])

    # Sort df_events by CHARTTIME; only keep the last value of a variable per
    # timestamp
    timeseries = df_events[['CHARTTIME', 'VARIABLE', 'VALUE', 'ITEMID']] \
            .sort_values(by=['CHARTTIME', 'VARIABLE', 'VALUE'], axis=0) \
            .drop_duplicates(subset=['CHARTTIME', 'VARIABLE'], keep='last')

    # Only keep first birth weight
    timeseries = timeseries[~(timeseries.duplicated(['ITEMID'], keep='first') &
        timeseries.ITEMID.isin([3723, 4183]))]

    # Pivot the dataframe s.t. the column names are the variables
    timeseries = timeseries.pivot(index='CHARTTIME', columns='VARIABLE',
            values='VALUE').sort_index(axis=0).reset_index()

    # Make sure that the timeseries contains all variables
    for v in variables:
        if v not in timeseries:
            timeseries[v] = np.nan

    # Make sure that if Weight and Height values are present, that the first
    # first row in the timeseries contain a value
    timeseries.Weight.iloc[0] = get_first_valid_value(timeseries, 'Weight')
    timeseries.Height.iloc[0] = get_first_valid_value(timeseries, 'Height')

    # Add GA weeks to timeseries
    ga_days = df_admit.iloc[0].GA_DAYS
    timeseries['Gestational Age -- Weeks'] = \
            timeseries['CHARTTIME'].apply(lambda x:
                    compute_ga_weeks_for_charttime(x, intime, ga_days))

    # Add target LOS to timeseries
    los_hours = df_admit.iloc[0].LOS_HOURS
    timeseries['TARGET'] = \
            timeseries['CHARTTIME'].apply(lambda x:
                    los_hours_to_target(compute_remaining_los(
                        x, intime, los_hours)))

    return timeseries


def main(args):
    verbose, subjects_path = args.verbose, args.subjects_path
    subject_directories = os.listdir(subjects_path)
    subject_directories = set(filter(lambda x: str.isdigit(x),
        subject_directories))
    tot_subjects = len(subject_directories)
    removed_subjects, tot_events, tot_events_kept = 0, 0, 0

    for i, subject_dir in enumerate(tqdm(subject_directories)):
        # Read the events dataframe
        df_events = pd.read_csv(os.path.join(subjects_path, str(subject_dir),
            'events.csv'))

        # Read the admission dataframe
        df_admit = pd.read_csv(os.path.join(subjects_path, str(subject_dir),
            'admission.csv'))

        # Create the timeseries
        timeseries = create_timeseries(df_events, df_admit)

        # Write timeseries to timeseries.csv if not empty, remove otherwise
        if not timeseries.empty:
            timeseries.to_csv(os.path.join(subjects_path, str(subject_dir),
                'timeseries.csv'), index=False)
        else:
            try:
                shutil.rmtree(os.path.join(subjects_path, str(subject_dir)))
                removed_subjects += 1
            except OSError as e:
                print (f'Error: {e.filename} - {e.strerror}.')

    if verbose:
        print(f'Of the initial {tot_subjects} subjects, ' \
                f'{tot_subjects-removed_subjects} remain that have ' \
                f'non-empty time-series.\n')

if __name__ == '__main__':
    main(parse_cl_args())

