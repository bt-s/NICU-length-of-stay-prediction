#!/usr/bin/python3

"""validate_events_and_notes.py

Script to validate the clinical events and notes extracted with
extract_event_data_per_subject.py.
"""

__author__ = "Bas Straathof"

import argparse, csv, os, shutil

from sys import argv
from tqdm import tqdm
from datetime import datetime, timedelta

import pandas as pd


def parse_cl_args():
    """Parses CL arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-sp', '--subjects-path', type=str, default='data/',
            help='Path to subject directories.')
    parser.add_argument('-op', '--output-path', type=str,
            default='validation_statistics.csv',
            help='Path to the output file.')
    parser.add_argument('-v', '--verbose', type=int,
            help='Level of verbosity in console output.', default=1)

    return parser.parse_args(argv[1:])


def validate_events_and_notes(df_stay, df_events, df_notes, stats):
    if list(stats.keys()) != ['events_tot_nb_events', 'events_no_value',
            'events_no_charttime', 'events_incorrect_charttime',
            'events_no_hadm_id_and_icustay_id', 'events_incorrect_hadm_id',
            'events_incorrect_icustay_id', 'events_final_nb_events',
            'notes_tot_nb_notes', 'notes_no_text', 'notes_incorrect_charttime',
            'notes_incorrect_hadm_id', 'notes_final_nb_notes']:
        raise ValueError('The keys of stats must be: events_tot_nb_events, ' +
                'events_no_value, events_no_charttime, ' +
                'events_incorrect_charttime, ' +
                'events_no_hadm_id_and_icustay_id, events_incorrect_hadm_id, ' +
                'events_incorrect_icustay_id, events_final_nb_events, ' +
                'notes_tot_nb_notes, notes_note_text, ' +
                'notes_incorrect_charttime, notes_incorrect_hadm_id, ' +
                'notes_final_nb_notes')

    tot_events, tot_notes = len(df_events), len(df_notes)

    # Make sure that the charttime field is datatime
    df_events.CHARTTIME = pd.to_datetime(df_events.CHARTTIME)
    df_notes.CHARTTIME = pd.to_datetime(df_notes.CHARTTIME)

    # Obtain the HADM_ID and ICUSTAY_ID from df_stay
    hadm_id = df_stay.HADM_ID.tolist()[0]
    icustay_id = df_stay.ICUSTAY_ID.tolist()[0]

    # Obtain INTIME and OUTTIME from df_stay and round down to hour
    date_format = "%Y-%m-%d %H:%M:%S"
    intime = df_stay.INTIME.tolist()[0]
    intime = datetime.strptime(intime, date_format)
    intime = intime.replace(microsecond=0, second=0, minute=0)
    outtime = df_stay.OUTTIME.tolist()[0]
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

    stats['events_tot_nb_events'] += tot_events
    stats['events_no_value'] += no_value
    stats['events_no_charttime'] += no_charttime
    stats['events_incorrect_charttime'] += incorrect_charttime
    stats['events_no_hadm_id_and_icustay_id'] += no_hadm_and_icustay
    stats['events_incorrect_hadm_id'] += wrong_hadm
    stats['events_incorrect_icustay_id'] += wrong_icustay
    stats['events_final_nb_events'] += len(df_events)
    stats['notes_tot_nb_notes'] += tot_notes
    stats['notes_no_text'] += no_text
    stats['notes_incorrect_charttime'] += incorrect_charttime
    stats['notes_incorrect_hadm_id'] += incorrect_hadm_id
    stats['notes_final_nb_notes'] += len(df_notes)

    return df_events, df_notes, stats


def main(args):
    subjects_path = args.subjects_path
    subject_directories = os.listdir(subjects_path)
    subject_directories = set(filter(lambda x: str.isdigit(x),
        subject_directories))
    tot_subjects = len(subject_directories)
    removed_subjects = 0

    stats = {'events_tot_nb_events': 0, 'events_no_value': 0,
            'events_no_charttime': 0, 'events_incorrect_charttime': 0,
            'events_no_hadm_id_and_icustay_id': 0,
            'events_incorrect_hadm_id': 0, 'events_incorrect_icustay_id': 0,
            'events_final_nb_events': 0, 'notes_tot_nb_notes': 0,
            'notes_no_text': 0, 'notes_incorrect_charttime': 0,
            'notes_incorrect_hadm_id': 0, 'notes_final_nb_notes': 0}

    for i, sd in enumerate(tqdm(subject_directories)):
        # Read the stay data for current subject
        df_stay = pd.read_csv(os.path.join(subjects_path, sd,
            'stay.csv'))

        # Assert the data frame only contains valid ICUSTAY_ID and HADM_ID
        assert(not df_stay.ICUSTAY_ID.isnull().any() and
                not df_stay.HADM_ID.isnull().any() and
                len(df_stay == 1))

        # Read the events for the current subject
        df_events = pd.read_csv(os.path.join(subjects_path, sd, 'events.csv'))

        # Read the notes for the current subject
        df_notes = pd.read_csv(os.path.join(subjects_path, sd, 'notes.csv'))

        # Validate events and notes
        df_events, df_notes, stats = validate_events_and_notes(df_stay,
                df_events, df_notes, stats)

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

    # Write results to the file
    with open(args.output_path, 'w') as wf:
        writer = csv.writer(wf)
        for key, value in stats.items():
            writer.writerow([key, value])

    if args.verbose:
        for k, v in stats.items():
            print(k+':', v)
        print(f'From the initial {tot_subjects} subjects, ' \
                f'{tot_subjects-removed_subjects} remain that have ' \
                'events associated with them.')


if __name__ == '__main__':
    main(parse_cl_args())

