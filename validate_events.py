import argparse
import os
import shutil

from sys import argv
from tqdm import tqdm
from datetime import datetime, timedelta

import pandas as pd


def parse_cl_args():
    """Parses CL arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-sp', '--subjects-path', type=str, default='data/',
            help='Path to subject directories.')
    parser.add_argument('-v', '--verbose', type=int,
            help='Level of verbosity in console output.', default=1)

    return parser.parse_args(argv[1:])


def validate_events(df_admission, df_events, stats):
    if list(stats.keys()) != ['tot_nb_events', 'no_charttime',
            'incorrect_charttime', 'no_hadm_id_and_icustay_id',
            'incorrect_hadm_id', 'incorrect_icustay_id', 'final_nb_events']:
        raise ValueError('The keys of stats must be: tot_nb_events, ' +
                'no_charttime, incorrect_charttime, ' +
                'no_hadm_id_and_icustay_id, incorrect_hadm_id, ' +
                'incorrect_icustay_id, final_nb_events')
    tot_events = len(df_events)

    # Make sure that the charttime field is datatime
    df_events.CHARTTIME = pd.to_datetime(df_events.CHARTTIME)

    # Obtain the HADM_ID and ICUSTAY_ID from df_admission
    hadm_id = df_admission.HADM_ID.tolist()[0]
    icustay_id = df_admission.ICUSTAY_ID.tolist()[0]

    # Obtain INTIME and OUTTIME from df_admission and round down to hour
    date_format = "%Y-%m-%d %H:%M:%S"
    intime = df_admission.INTIME.tolist()[0]
    intime = datetime.strptime(intime, date_format)
    intime = intime.replace(microsecond=0, second=0, minute=0)
    outtime = df_admission.OUTTIME.tolist()[0]
    outtime = datetime.strptime(outtime, date_format)
    outtime = outtime.replace(microsecond=0, second=0, minute=0)

    # Drop events without a charttime
    df_events = df_events[df_events.CHARTTIME.notna()]
    no_charttime = tot_events - len(df_events)

    # Drop events that fall outside the intime-outtime interval
    mask = (intime <= df_events.CHARTTIME) & (df_events.CHARTTIME < outtime)
    df_events = df_events.loc[mask]
    incorrect_charttime = tot_events - len(df_events) - no_charttime

    # Drop events for which both HADM_ID and ICUSTAY_ID are empty
    df_events = df_events.dropna(how='all', subset=['HADM_ID',
        'ICUSTAY_ID'])
    no_hadm_and_icustay = tot_events - len(df_events) - no_charttime \
            - incorrect_charttime

    # Keep events without HADM_ID but with correct ICUSTAY_ID as well as
    # events without ICUSTAY_ID but with correct HADM_ID
    df_events = df_events.fillna(value={'HADM_ID': hadm_id,
        'ICUSTAY_ID': icustay_id})

    # Drop events with wrong HADM_ID
    df_events = df_events.loc[df_events.HADM_ID == hadm_id]
    wrong_hadm = tot_events - len(df_events) - no_charttime \
            - incorrect_charttime - no_hadm_and_icustay

    # Drop events with wrong ICUSTAY_ID
    df_events = df_events.loc[df_events.ICUSTAY_ID == icustay_id]
    wrong_icustay = tot_events - len(df_events) - no_charttime \
            - incorrect_charttime - no_hadm_and_icustay - wrong_hadm

    stats['tot_nb_events'] += tot_events
    stats['no_charttime'] += no_charttime
    stats['incorrect_charttime'] += incorrect_charttime
    stats['no_hadm_id_and_icustay_id'] += no_hadm_and_icustay
    stats['incorrect_hadm_id'] += wrong_hadm
    stats['incorrect_icustay_id'] += wrong_icustay
    stats['final_nb_events'] += len(df_events)

    return df_events, stats


def main(args):
    verbose, subjects_path = args.verbose, args.subjects_path
    subject_directories = os.listdir(subjects_path)
    subject_directories = set(filter(lambda x: str.isdigit(x),
        subject_directories))
    tot_subjects = len(subject_directories)
    removed_subjects = 0

    stats = {'tot_nb_events': 0, 'no_charttime': 0, 'incorrect_charttime': 0,
            'no_hadm_id_and_icustay_id': 0, 'incorrect_hadm_id': 0,
            'incorrect_icustay_id': 0, 'final_nb_events': 0}

    for i, sd in enumerate(tqdm(subject_directories)):
        # Read the admission for current subject
        df_admission = pd.read_csv(os.path.join(subjects_path, sd,
            'admission.csv'))

        # Assert the data frame only contains valid ICUSTAY_ID and HADM_ID
        assert(not df_admission.ICUSTAY_ID.isnull().any() and
                not df_admission.HADM_ID.isnull().any() and
                len(df_admission == 1))

        # Read the events for the current subject
        df_events = pd.read_csv(os.path.join(subjects_path, sd, 'events.csv'))

        # Validate events
        df_events, stats = validate_events(df_admission, df_events, stats)

        if not df.empty:
            # Write df_events to events.csv
            df_events.to_csv(os.path.join(subjects_path, sd, 'events.csv'))
        else:
            # Remove the folder
            try:
                shutil.rmtree(os.path.join(subjects_path, sd))
                removed_subjects += 1
            except OSError as e:
                print (f'Error: {e.filename} - {e.strerror}.')

    if verbose:
        for k, v in stats.items():
            print(k+':', v)
        print(f'From the initial {tot_subjects} subjects, ' \
                f'{tot_subjects-removed_subjects} remain that have ' \
                'events associated with them.')


if __name__ == '__main__':
    main(parse_cl_args())

