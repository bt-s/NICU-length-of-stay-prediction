#!/usr/bin/python3

"""process_mimic3_tables.py

Script to extract data of newborn patients from the MIMIC-III CSVs.  For each
patient an output directory is created, in which the data are stored in
stay.csv (admisison info), events.csv (clinical events) and notes.csv (clinical
notes).
"""

__author__ = "Bas Straathof"

import argparse, os, pickle, sys

import pandas as pd
from tqdm import tqdm
from itertools import repeat
import multiprocessing as mp

from ..utils.mimic_readers import MimicNICUReaders
from ..utils.utils import get_subject_dirs
from ..utils.reg_exps import reg_exps
from ..utils.preprocessing_utils import filter_on_newborns, \
        extract_gest_age_from_note, transfer_filter, process_notes, \
        read_and_split_table_by_subject, validate_events_and_notes, \
        validate_events_and_notes_per_subject


def parse_cl_args():
    """Parses CL arguments"""
    parser = argparse.ArgumentParser(
            description='Extract data from the MIMIC-III CSVs.')
    parser.add_argument('-ip', '--input-path', type=str,
            help='Path to MIMIC-III CSV files.', default='../../mimic/')
    parser.add_argument('-op', '--output-path', type=str,
            default='data/', help='Path to desired output directory.')
    parser.add_argument('-v', '--verbose', type=int,
            help='Info in console output (0 or 1).', default=1)

    return parser.parse_args(sys.argv[1:])


def main(args):
    mimic_iii_path, output_path  = args.input_path, args.output_path

    v_print = print if args.verbose else lambda *a, **k: None

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    reader = MimicNICUReaders(mimic_iii_path, args.verbose)

    df = reader.read_admissions_table()
    df_icu = reader.read_icustays_table()
    df_pat = reader.read_patients_table()
    df_lab = reader.read_labevents_table()

    df = df_icu.merge(df, how='inner', on=['SUBJECT_ID', 'HADM_ID'])
    v_print(f'Filtered NICU admissions -- with admission ' \
            f'information: {df.shape[0]}')

    df = df.merge(df_pat, how='inner', on='SUBJECT_ID')
    v_print(f'Filtered NICU admissions -- with patient information: '
            f'{df.shape[0]}')

    df = filter_on_newborns(df)
    v_print(f'Filtered NICU admissions -- newborn only {df.shape[0]}')

    df = df[df.SUBJECT_ID.isin(df_lab.SUBJECT_ID)]
    v_print(f'Filtered NICU admissions -- with associated ' \
            f'lab events: {df.shape[0]}')

    df_notes = reader.read_noteevents_table()

    # Filter df_notes on subjects and admissions in df
    df_notes = df_notes[df_notes.SUBJECT_ID.isin(df.SUBJECT_ID)]
    df_notes = df_notes[df_notes.HADM_ID.isin(df.HADM_ID)]

    # Filter on subjects that have notes associated with them
    df = df[df.SUBJECT_ID.isin(df_notes.SUBJECT_ID)]
    v_print(f'Filtered NICU admissions -- with associated ' \
            f'notes: {df.shape[0]}')

    v_print('...extract GA from notes and remove admissions with a capacity ' \
            'related transfer...')
    df_ga, cap_trans_set = process_notes(df_notes, reg_exps)

    df = df.merge(df_ga, how='inner', on='SUBJECT_ID')
    v_print(f'Filtered NICU admissions -- with GA: {df.shape[0]}')

    # Filter out admissions with capacity-related transfers
    df = df[~df.SUBJECT_ID.isin(cap_trans_set)]
    v_print(f'Filtered NICU admissions -- without capacity ' \
            f'related transfers: {df.shape[0]}')

    v_print(f'{df.HOSPITAL_EXPIRE_FLAG.sum()}/{df.shape[0]} newborns in df ' \
            'died during their NICU admission.')

    v_print('...split admissions by subject...')
    tot_nb_subjects = len(df.SUBJECT_ID.unique())

    # Write admission information to directory per subject
    subjects_to_keep = set()
    for i, (ix, row) in enumerate(tqdm(df.iterrows(), total=df.shape[0])):
        subject_f = os.path.join(output_path, str(row.SUBJECT_ID))
        subjects_to_keep.add(row.SUBJECT_ID)

        if not os.path.exists(subject_f):
            os.makedirs(subject_f)

        df.loc[df.SUBJECT_ID == row.SUBJECT_ID].to_csv(
            os.path.join(subject_f, 'stay.csv'), index=False)

    # Read and split MIMIC-III event tables per subject
    # Using multiprocessing to read the tables simultaneously
    table_names = ['chartevents', 'labevents', 'noteevents']

    with mp.Pool() as p:
        p.starmap(read_and_split_table_by_subject, zip(repeat(mimic_iii_path),
            table_names, repeat(output_path), repeat(subjects_to_keep),
            repeat(args.verbose), range(len(table_names))))

    # Validate the events and notes
    subject_directories = get_subject_dirs(output_path)
    validate_events_and_notes(subject_directories)


if __name__ == '__main__':
    main(parse_cl_args())

