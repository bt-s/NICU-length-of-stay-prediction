#!/usr/bin/python3

"""extract_newborns.py - Script to extract data of newborn patients from the
                         MIMIC-III CSVs.

As part of my Master's thesis at KTH Royal Institute of Technology.
"""

__author__ = "Bas Straathof"

import argparse
from sys import argv

from utils import *


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
    try:
        os.makedirs(args.output_path)
    except:
        pass

    verbose, mimic_iii_path  = args.verbose, args.input_path

    if verbose: print('Reading ICUSTAYS table...')
    df, tot_icu_admit, tot_nicu_admit = read_icustays_table(mimic_iii_path)

    if verbose: print('Reading ADMISSIONS table...')
    df_admit, tot_admit, nb_admit = read_admissions_table(mimic_iii_path)

    # Merge admission information into dataframe
    df = df.merge(df_admit, how='inner', on=['SUBJECT_ID', 'HADM_ID'])

    print('Reading PATIENTS table...')
    df_pat = read_patients_table(mimic_iii_path)

    # Merge patients information into dataframe
    df = df.merge(df_pat, how='inner', on='SUBJECT_ID')

    # Only keep the first admission for each patient
    if verbose: print("Removing all but the first admission for each patient...")
    df = filter_on_first_admission(df)

    if verbose:
        print(f'Extracted first complete neonatal ICU admissions: {len(df)}')

    if verbose:
        print(f'Total admissions identified: {tot_admit}\n' \
                f'Newborn admissions identified: {nb_admit}\n' \
                f'Total ICU admissions identified: {tot_icu_admit}\n' \
                f'Neonatal ICU admissions identified: {tot_nicu_admit}\n' \
                f'Total first complete neonatal ICU admissions: {len(df)}')

if __name__ == '__main__':
    main(parse_cl_args())

