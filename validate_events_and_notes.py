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


def validate_events_and_notes(df_admission, df_events, df_notes, stats_events,
        stats_notes):
    if list(stats_events.keys()) != ['tot_nb_events', 'no_value',
            'no_charttime', 'incorrect_charttime', 'no_hadm_id_and_icustay_id',
            'incorrect_hadm_id', 'incorrect_icustay_id', 'final_nb_events']:
        raise ValueError('The keys of stats must be: tot_nb_events, ' +
                'no_value, no_charttime, incorrect_charttime, ' +
                'no_hadm_id_and_icustay_id, incorrect_hadm_id, ' +
                'incorrect_icustay_id, final_nb_events')

    if list(stats_notes.keys()) != ['tot_nb_notes', 'no_text',
            'incorrect_charttime', 'incorrect_hadm_id', 'final_nb_notes']:
        raise ValueError('The keys of stats must be: tot_nb_notes, note_text, ' +
                'incorrect_charttime, incorrect_hadm_id, final_nb_notes')

    tot_events, tot_notes = len(df_events), len(df_notes)

    # Make sure that the charttime field is datatime
    df_events.CHARTTIME = pd.to_datetime(df_events.CHARTTIME)
    df_notes.CHARTTIME = pd.to_datetime(df_notes.CHARTTIME)

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

    # Drop events without a value
    df_events = df_events[df_events.VALUE.notna()]
    no_value = tot_events - len(df_events) - no_charttime

    # Make sure that if VALUEOM is not present that it is an empty string
    df_events.VALUEUOM= df_events.VALUEUOM.fillna('').astype(str)

    # Drop events that fall outside the intime-outtime interval
    mask = (intime <= df_events.CHARTTIME) & (df_events.CHARTTIME < outtime)
    df_events = df_events.loc[mask]
    incorrect_charttime = tot_events - len(df_events) - no_value - no_charttime

    # Drop events for which both HADM_ID and ICUSTAY_ID are empty
    df_events = df_events.dropna(how='all', subset=['HADM_ID',
        'ICUSTAY_ID'])
    no_hadm_and_icustay = tot_events - len(df_events) - no_value \
            - no_charttime - incorrect_charttime

    # Keep events without HADM_ID but with correct ICUSTAY_ID as well as
    # events without ICUSTAY_ID but with correct HADM_ID
    df_events = df_events.fillna(value={'HADM_ID': hadm_id,
        'ICUSTAY_ID': icustay_id})

    # Drop events with wrong HADM_ID
    df_events = df_events.loc[df_events.HADM_ID == hadm_id]
    wrong_hadm = tot_events - len(df_events) - no_value - no_charttime \
            - incorrect_charttime - no_hadm_and_icustay

    # Drop events with wrong ICUSTAY_ID
    df_events = df_events.loc[df_events.ICUSTAY_ID == icustay_id]
    wrong_icustay = tot_events - len(df_events) - no_value - no_charttime \
            - incorrect_charttime - no_hadm_and_icustay - wrong_hadm

    # Drop notes without a charttime
    df_notes = df_notes[df_notes.CHARTTIME.notna()]

    # Drop notes that fall outside the intime-outtime interval
    mask = (intime <= df_notes.CHARTTIME) & (df_notes.CHARTTIME < outtime)
    df_notes = df_notes.loc[mask]
    incorrect_charttime = tot_notes - len(df_notes)

    # Drop notes without a text
    df_notes = df_notes[(df_notes.TEXT.notna())]
    no_text = tot_notes - len(df_notes) - incorrect_charttime

    # Drop notes for which HADM_ID
    df_notes = df_notes[df_notes.HADM_ID.notna()]

    # Drop notes with wrong HADM_ID
    df_notes = df_notes.loc[df_notes.HADM_ID == hadm_id]
    incorrect_hadm_id = tot_notes - len(df_notes) - no_text - \
            incorrect_charttime

    stats_events['tot_nb_events'] += tot_events
    stats_events['no_value'] += no_value
    stats_events['no_charttime'] += no_charttime
    stats_events['incorrect_charttime'] += incorrect_charttime
    stats_events['no_hadm_id_and_icustay_id'] += no_hadm_and_icustay
    stats_events['incorrect_hadm_id'] += wrong_hadm
    stats_events['incorrect_icustay_id'] += wrong_icustay
    stats_events['final_nb_events'] += len(df_events)

    stats_notes['tot_nb_notes'] += tot_notes
    stats_notes['no_text'] += no_text
    stats_notes['incorrect_charttime'] += incorrect_charttime
    stats_notes['incorrect_hadm_id'] += incorrect_hadm_id
    stats_notes['final_nb_notes'] += len(df_notes)

    return df_events, df_notes, stats_events, stats_notes


def main(args):
    verbose, subjects_path = args.verbose, args.subjects_path
    subject_directories = os.listdir(subjects_path)
    subject_directories = set(filter(lambda x: str.isdigit(x),
        subject_directories))
    tot_subjects = len(subject_directories)
    removed_subjects = 0

    stats_events = {'tot_nb_events': 0, 'no_value': 0, 'no_charttime': 0,
            'incorrect_charttime': 0, 'no_hadm_id_and_icustay_id': 0,
            'incorrect_hadm_id': 0, 'incorrect_icustay_id': 0,
            'final_nb_events': 0}

    stats_notes = {'tot_nb_notes': 0, 'no_text': 0, 'incorrect_charttime': 0,
            'incorrect_hadm_id': 0, 'final_nb_notes': 0}


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

        # Read the notes for the current subject
        df_notes = pd.read_csv(os.path.join(subjects_path, sd, 'notes.csv'))

        # Validate events and notes
        df_events, df_notes, stats_events, stats_notes = \
                validate_events_and_notes(df_admission, df_events, df_notes,
                stats_events, stats_notes)

        if (not df_events.empty) and (not df_notes.empty) :
            # Write df_events to events.csv
            df_events.to_csv(os.path.join(subjects_path, sd, 'events.csv'),
                    index=False)

            # Write df_notes to notes.csv
            df_notes.to_csv(os.path.join(subjects_path, sd, 'notes.csv'),
                    index=False)
        else:
            # Remove the folder
            try:
                shutil.rmtree(os.path.join(subjects_path, sd))
                removed_subjects += 1
            except OSError as e:
                print (f'Error: {e.filename} - {e.strerror}.')

    if verbose:
        for k, v in stats_events.items():
            print(k+':', v)
        for k, v in stats_notes.items():
            print(k+':', v)
        print(f'From the initial {tot_subjects} subjects, ' \
                f'{tot_subjects-removed_subjects} remain that have ' \
                'events associated with them.')


if __name__ == '__main__':
    main(parse_cl_args())
