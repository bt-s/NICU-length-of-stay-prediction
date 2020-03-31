#!/usr/bin/python3

"""extract_nicu_stay_data_per_subjects.py

Script to extract data of newborn patients from the MIMIC-III CSVs.  For each
patient an output directory is created, in which the data is stored in a file
called stay.csv.
"""

__author__ = "Bas Straathof"

import argparse, os, pickle

import pandas as pd
from tqdm import tqdm

from sys import argv

from csv_readers import read_admissions_table, read_icustays_table, \
        read_patients_table, read_noteevents_table, read_labevents_table
from utils import filter_on_newborns, extract_gest_age_from_note, \
        transfer_filter
from reg_exps import reg_exps


def parse_cl_args():
    """Parses CL arguments"""
    parser = argparse.ArgumentParser(
            description='Extract data from the MIMIC-III CSVs.')
    parser.add_argument('-ip', '--input-path', type=str,
            help='Path to MIMIC-III CSV files.')
    parser.add_argument('-op', '--output-path', type=str,
            default='data/', help='Path to desired output directory.')
    parser.add_argument('-v', '--verbose', type=int,
            help='Info in console output (0 or 1).', default=1)

    return parser.parse_args(argv[1:])


def main(args):
    mimic_iii_path, output_path  = args.input_path, args.output_path
    verbose = args.verbose

    v_print = print if verbose else lambda *a, **k: None

    try:
        os.makedirs(output_path)
    except:
        pass

    v_print('...read ADMISSIONS table...')
    df = read_admissions_table(mimic_iii_path)
    v_print(f'Total admissions identified: {df.shape[0]}')

    # Data set filtered on NICU admissions
    df = df[df.ADMISSION_TYPE == 'NEWBORN']
    v_print(f'Total NICU admissions identified: {df.shape[0]}\n' \
            f'Total unique NICU patients identified: {df.SUBJECT_ID.nunique()}')

    # Data set filtered on newborn admissions
    df = df[df.DIAGNOSIS == 'NEWBORN']

    # Make sure that there are no duplicate SUBJECT_IDs in df
    v_print(f'Total newborn admissions identified: {df.shape[0]}\n' \
            f'Total unique newborn patients identified: ' \
            f'{df.SUBJECT_ID.nunique()}')

    # Only keep admissions with associated chartevents
    df = df[df.HAS_CHARTEVENTS_DATA == 1]
    v_print(f'Filtered newborn admissions -- with chart events: {df.shape[0]}')

    v_print('...read ICUSTAYS table...')
    df_icu = read_icustays_table(mimic_iii_path)
    v_print(f'Total ICU stays identified: {df_icu.shape[0]}')

    # Filter on neonatal ICU
    df_icu = df_icu [df_icu.FIRST_CAREUNIT == 'NICU']
    v_print(f'Total NICU stays identified: {df_icu.shape[0]}')

    # Only keep NICU stays without transfers
    df_icu = df_icu.loc[(df_icu.FIRST_WARDID == df_icu.LAST_WARDID) &
            (df_icu.FIRST_CAREUNIT == df_icu.LAST_CAREUNIT)]
    v_print(f'Filtered NICU stays -- without transfers: {df_icu.shape[0]}')

    # Only keep the first stay
    df_first_admin = df_icu[['SUBJECT_ID', 'INTIME']].groupby(
            'SUBJECT_ID').min().reset_index()
    df_icu = df_icu[df_icu.INTIME.isin(df_first_admin.INTIME)]
    v_print(f'Filtered NICU stays -- first stay: {df_icu.shape[0]}')

    # Remove admissions with undefined LOS
    df_icu = df_icu[df_icu.LOS.isnull() == False]
    v_print(f'Filtered NICU stays -- defined LOS: {df.shape[0]}')

    # Remove admission shorter than four hours
    df_icu = df_icu[df_icu.LOS >= 1/6]
    v_print(f'Filtered NICU stays -- longer than four hours: {df_icu.shape[0]}')

    # Create rounded LOS_HOURS variable
    df_icu['LOS_HOURS'] = round(df_icu.LOS * 24, 0).astype('int')

    v_print('...read PATIENTS table...')
    df_pat = read_patients_table(mimic_iii_path)
    v_print(f'Total patients identified: {df_pat.shape[0]}')

    v_print('...merge admission and ICU information...')
    df = df_icu.merge(df, how='inner', on=['SUBJECT_ID', 'HADM_ID'])
    v_print(f'Filtered NICU admissions -- with admission ' \
            f'information: {df.shape[0]}')

    v_print('...merge patients information into dataframe...')
    df = df.merge(df_pat, how='inner', on='SUBJECT_ID')
    v_print(f'Filtered NICU admissions -- with patient information: {df.shape[0]}')

    v_print("...remove admissions of non-newborn patients...")
    df = filter_on_newborns(df)
    v_print(f'Filtered NICU admissions -- newborn only {df.shape[0]}')

    v_print('...read LABEVENTS table...')
    df_lab = read_labevents_table(mimic_iii_path)

    # Filter df_lab on subjects and admissions in df
    df_lab = df_lab[df_lab.SUBJECT_ID.isin(df.SUBJECT_ID)]
    df_lab = df_lab[df_lab.HADM_ID.isin(df.HADM_ID)]

    # Filter on subjects that have lab events associated with them
    df = df[df.SUBJECT_ID.isin(df_lab.SUBJECT_ID)]
    v_print(f'Filtered NICU admissions -- with associated ' \
            f'lab events: {df.shape[0]}')

    v_print('...read NOTEEVENTS table...')
    df_notes = read_noteevents_table(mimic_iii_path)

    # Filter df_notes on subjects and admissions in df
    df_notes = df_notes[df_notes.SUBJECT_ID.isin(df.SUBJECT_ID)]
    df_notes = df_notes[df_notes.HADM_ID.isin(df.HADM_ID)]

    # Filter on subjects that have notes associated with them
    df = df[df.SUBJECT_ID.isin(df_notes.SUBJECT_ID)]
    v_print(f'Filtered NICU admissions -- with associated ' \
            f'notes: {df.shape[0]}')

    v_print('...extract GA from notes and remove admissions with a capacity ' \
            'related transfer...')
    df_ga = pd.DataFrame(columns=['SUBJECT_ID', 'GA_MATCH', 'GA_DAYS',
        'GA_WEEKS_ROUND'])

    gest_age_set, cap_trans_set = set(), set()
    for ix, row in tqdm(df_notes.iterrows(), total=df.shape[0]):
        m_ga, m_ct = None, None # Default is that no match will be found

        # Look for GA
        if row.SUBJECT_ID not in gest_age_set:
            m_ga, d, w = extract_gest_age_from_note(row.TEXT, reg_exps)
            if m_ga:
                gest_age_set.add(row.SUBJECT_ID)
                df_ga.loc[ix] = [row.SUBJECT_ID] + [m_ga] + [d] + [w]

        # Look for capacity-related transfers
        if row.SUBJECT_ID not in cap_trans_set:
            m_ct = transfer_filter(row.TEXT, reg_exps)
            if m_ct:
                cap_trans_set.add(row.SUBJECT_ID)

    v_print(f'Total GAs identified: {len(gest_age_set)}\nTotal ' \
            f'capacity-related admissions identified: {len(cap_trans_set)}')

    v_print('...merge GA information into dataframe...')
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

    for i, (ix, row) in enumerate(tqdm(df.iterrows(), total=df.shape[0])):
        if verbose and i % 250 == 0:
            print(f'Creating file for subject {i}/{tot_nb_subjects}')

        subject_f = os.path.join(output_path, str(row.SUBJECT_ID))

        try:
            os.makedirs(subject_f)
        except:
            pass

        df.loc[df.SUBJECT_ID == row.SUBJECT_ID].to_csv(
            os.path.join(subject_f, 'stay.csv'), index=False)


if __name__ == '__main__':
    main(parse_cl_args())

