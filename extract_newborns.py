#!/usr/bin/python3

"""extract_newborns.py - Script to extract data of newborn patients from the
                         MIMIC-III CSVs.

As part of my Master's thesis at KTH Royal Institute of Technology.
"""

__author__ = "Bas Straathof"

import argparse, os, pickle
from sys import argv

from utils import *
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
            help='Level of verbosity in console output.', default=1)

    return parser.parse_args(argv[1:])


def main(args):
    verbose, mimic_iii_path  = args.verbose, args.input_path

    try:
        os.makedirs(args.output_path)
    except:
        pass

    # Read the ADMISSIONS table
    df_admit = read_admissions_table(mimic_iii_path, verbose)

    # Read the ICUSTAYS table
    df = read_icustays_table(mimic_iii_path,
            verbose)

    # Read the PATIENTS table
    df_pat = read_patients_table(mimic_iii_path, verbose)

    if verbose: print('...merge admission information into dataframe...')
    df = df.merge(df_admit, how='inner', on=['SUBJECT_ID', 'HADM_ID'])
    if verbose: print(f'Filtered NICU admissions -- with admission ' \
            f'information: {len(df)}')

    if verbose: print('...merge patients information into dataframe...')
    df = df.merge(df_pat, how='inner', on='SUBJECT_ID')
    if verbose: print(f'Filtered NICU admissions -- with patient ' \
            f'information: {len(df)}')

    if verbose: print("...remove admissions of non-newborn patients...")
    df = filter_on_newborns(df)
    if verbose: print(f'Filtered NICU admissions -- newborn only {len(df)}')

    # Read the LABEVENTS table
    df_lab = read_labevents_table(mimic_iii_path, verbose)

    # Filter df_lab on subjects and admissions in df
    df_lab = df_lab[df_lab['SUBJECT_ID'].isin(df['SUBJECT_ID'])]
    df_lab = df_lab[df_lab['HADM_ID'].isin(df['HADM_ID'])]

    # Filter on subjects that have lab events associated with them
    df = df[df['SUBJECT_ID'].isin(df_lab['SUBJECT_ID'])]
    if verbose: print(f'Filtered NICU admissions -- with associated ' \
            f'lab events: {len(df)}')

    # Read the NOTEEVENTS table
    df_notes = read_noteevents_table(mimic_iii_path, verbose)

    # Filter df_notes on subjects and admissions in df
    df_notes = df_notes[df_notes['SUBJECT_ID'].isin(df['SUBJECT_ID'])]
    df_notes = df_notes[df_notes['HADM_ID'].isin(df['HADM_ID'])]

    # Filter on subjects that have notes associated with them
    df = df[df['SUBJECT_ID'].isin(df_notes['SUBJECT_ID'])]
    if verbose: print(f'Filtered NICU admissions -- with associated ' \
            f'notes: {len(df)}')

    if verbose: print('...extract GA from notes and remove admissions with a ' +
            'capacity related transfer...')
    # Create a temporary dataframe to capture the GA from df_notes
    df_ga = pd.DataFrame(columns=['SUBJECT_ID', 'GA_MATCH', 'GA_DAYS',
        'GA_WEEKS_ROUND'])

    gest_age_set, cap_trans_set = set(), set()
    for ix, row in df_notes.iterrows():
        m_ga, m_ct = None, None # Default is that no match will be found

        # Look for GA
        if row.SUBJECT_ID not in gest_age_set:
            m_ga, d, w = extract_gest_age(row.TEXT, reg_exps)
            if m_ga:
                gest_age_set.add(row.SUBJECT_ID)
                df_ga.loc[ix] = [row.SUBJECT_ID] + [m_ga] + [d] + [w]

        # Look for capacity-related transfers
        if row.SUBJECT_ID not in cap_trans_set:
            m_ct = transfer_filter(row.TEXT, reg_exps)
            if m_ct:
                cap_trans_set.add(row.SUBJECT_ID)

    if verbose: print(f'Total GAs identified: {len(gest_age_set)}\n' +
            'Total capacity-related admissions identified: ' +
            f'{len(cap_trans_set)}')

    if verbose: print('...merge GA information into dataframe...')
    df = df.merge(df_ga, how='inner', on='SUBJECT_ID')
    if verbose: print(f'Filtered NICU admissions -- with GA: {len(df)}')

    # Filter out admissions with capacity-related transfers
    df = df[~df['SUBJECT_ID'].isin(cap_trans_set)]
    if verbose: print(f'Filtered NICU admissions -- without capacity ' \
            f'related transfers: {len(df)}')

    if verbose: print(f'{df.HOSPITAL_EXPIRE_FLAG.sum()}/{len(df)} newborns in '+
            'df died during their NICU admission.')

    if verbose: print(f'...creating targets for admissions...')
    df = set_targets(df)

    if verbose: print('...split admissions by subject...')
    split_admissions_by_subject(df, args.output_path, verbose=1)

    if verbose: print('...pickle dataframe...')
    df.to_pickle(os.path.join(args.output_path, 'subjects.pkl'))

    if verbose: print('...pickle list of subjects...')
    subjects = set(df.SUBJECT_ID.to_list())

    with open('data/subjects_list.pkl', 'wb') as f:
        pickle.dump(subjects, f)


if __name__ == '__main__':
    main(parse_cl_args())

