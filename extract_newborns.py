#!/usr/bin/python3

"""extract_newborns.py - Script to extract data of newborn patients from the
                         MIMIC-III CSVs.

As part of my Master's thesis at KTH Royal Institute of Technology.
"""

__author__ = "Bas Straathof"

import argparse
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
            help='Path to desired output directory.')
    parser.add_argument('-v', '--verbose', type=int,
            help='Level of verbosity in console output.', default=1)

    return parser.parse_args(argv[1:])


def main(args):
    verbose, mimic_iii_path  = args.verbose, args.input_path

    # Read the ICUSTAYS table
    df, tot_icu_admit, tot_nicu_admit = read_icustays_table(mimic_iii_path,
            verbose)

    # Read the ADMISSIONS table
    df_admit, tot_admit, nb_admit = read_admissions_table(mimic_iii_path,
            verbose)

    if verbose: print('Merge admission information into dataframe...')
    df = df.merge(df_admit, how='inner', on=['SUBJECT_ID', 'HADM_ID'])

    # Read the PATIENTS table
    df_pat = read_patients_table(mimic_iii_path, verbose)

    if verbose: print('Merge patients information into dataframe...')
    df = df.merge(df_pat, how='inner', on='SUBJECT_ID')

    if verbose: print("Remove all but the first admission for each patient...")
    df = filter_on_first_admission(df)

    if verbose:
        print(f'Extracted first complete neonatal ICU admissions: {len(df)}')

    # Read the NOTEEVENTS table
    df_notes = read_noteevents_table(mimic_iii_path, verbose)

    # Filter df_notes on subjects and admissions in df
    df_notes = df_notes[df_notes['SUBJECT_ID'].isin(df['SUBJECT_ID'])]
    df_notes = df_notes[df_notes['HADM_ID'].isin(df['HADM_ID'])]

    # Filter on subjects that have notes associated with them
    df = df[df['SUBJECT_ID'].isin(df_notes['SUBJECT_ID'])]

    if verbose:
        print(f'Total admissions identified: {tot_admit}\n' \
                f'Newborn admissions identified: {nb_admit}\n' \
                f'Total ICU admissions identified: {tot_icu_admit}\n' \
                f'Neonatal ICU admissions identified: {tot_nicu_admit}\n' \
                f'Total first complete neonatal ICU admissions: {len(df)}')

    if verbose: print('Extract GA from notes...')
    # Create a temporary dataframe to capture the GA from df_notes
    df_ga = pd.DataFrame(columns=['SUBJECT_ID', 'GA_MATCH', 'GA_DAYS',
        'GA_WEEKS_ROUND'])

    gest_age_set = set()
    for ix, row in df_notes.iterrows():
        m = None # Default is that no match will be found
        if row.SUBJECT_ID not in gest_age_set:
            m, d, w = extract_gest_age(row.TEXT, reg_exps)
        if m:
            gest_age_set.add(row.SUBJECT_ID)
            df_ga.loc[ix] = [row.SUBJECT_ID] + [m] + [d] + [w]
    if verbose: print(f'Total GAs identified: {len(gest_age_set)}')

    if verbose: print('Merge GA information into dataframe...')
    df = df.merge(df_ga, how='inner', on='SUBJECT_ID')

    if verbose: print('Pickle dataframe...')
    df.to_pickle(args.output_path)


if __name__ == '__main__':
    main(parse_cl_args())

