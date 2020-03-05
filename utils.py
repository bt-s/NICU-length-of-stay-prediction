#!/usr/bin/python3

"""utils.py - Utils for extracting data of newborn patients from the
              MIMIC-III CSVs.

As part of my Master's thesis at KTH Royal Institute of Technology.
"""

__author__ = "Bas Straathof"

import pandas as pd
import numpy as np
import os

from word2number import w2n


def read_icustays_table(mimic_iii_path):
    df = pd.read_csv(os.path.join(mimic_iii_path, 'ICUSTAYS.csv'))
    tot_icu_admit = len(df)

    # Keep neonatal ICU admissions
    df = df[df.FIRST_CAREUNIT == 'NICU']
    tot_nicu_admit = len(df)

    # Make sure that the time fields are datatime
    df.INTIME = pd.to_datetime(df.INTIME)
    df.OUTTIME = pd.to_datetime(df.OUTTIME)

    # Only keep neonatal ICU stays without transfers
    df = df.loc[(df.FIRST_WARDID == df.LAST_WARDID) &
            (df.FIRST_CAREUNIT == df.LAST_CAREUNIT)]

    # Only keep relevant columns
    df[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'LAST_CAREUNIT',
        'DBSOURCE', 'INTIME', 'OUTTIME', 'LOS']]

    return df, tot_icu_admit, tot_nicu_admit


def read_admissions_table(mimic_iii_path):
    df = pd.read_csv(os.path.join(mimic_iii_path, 'ADMISSIONS.csv'))
    tot_admit = len(df)

    # Keep newborns
    df = df[df.ADMISSION_TYPE == 'NEWBORN']
    nb_admit = len(df)

    # Keep relevant columns
    df = df[['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'ADMITTIME',
        'DISCHTIME', 'DEATHTIME', 'HOSPITAL_EXPIRE_FLAG',
        'HAS_CHARTEVENTS_DATA']]

    # Make sure that the time fields are datatime
    df.ADMITTIME = pd.to_datetime(df.ADMITTIME)
    df.DISCHTIME = pd.to_datetime(df.DISCHTIME)
    df.DEATHTIME = pd.to_datetime(df.DEATHTIME)

    return df, tot_admit, nb_admit


def read_patients_table(mimic_iii_path):
    df = pd.read_csv(os.path.join(mimic_iii_path, 'PATIENTS.csv'))

    # Only keep relevant columns
    df = df[['SUBJECT_ID', 'GENDER', 'DOB', 'DOD']]

    # Make sure that the time fields are datatime
    df.DOB = pd.to_datetime(df.DOB)
    df.DOD = pd.to_datetime(df.DOD)

    return df


def read_noteevents_table(mimic_iii_path):
    df = pd.read_csv(os.path.join(mimic_iii_path, 'NOTEEVENTS.csv'))

    # Only keep relevant columns
    df = df[['SUBJECT_ID', 'HADM_ID', 'CHARTDATE', 'CHARTTIME', 'CATEGORY',
        'DESCRIPTION', 'ISERROR', 'TEXT']]

    return df


def filter_on_first_admission(df):
    df_first_admin = df[['SUBJECT_ID', 'INTIME']].groupby(
            'SUBJECT_ID').min().reset_index()
    df = df[df['INTIME'].isin(df_first_admin['INTIME'])]

    return df


def extract_from_cga_match(match_cga, reg_exps):
    # Extract matched string
    match_str_cga = match_cga.group(0)

    # Split the match of the CGA to be able to compute the number of days
    # and weeks
    match_str = reg_exps['re_splitter'].split(match_str_cga)
    match_str_dol = match_str[0]
    match_str_cga = match_str[2]
    # Extract the days of life
    dol = int(reg_exps['re_dol'].search(match_str_dol).group(0))

    # Extract the days part of the correct gestational age
    if reg_exps['re_dd_d'].findall(match_str_cga):
        days_cga = int(reg_exps['re_dd_d'].search(match_str_cga)
                .group(0)[-1])
    elif reg_exps['re_anon_dd_p'].findall(match_str_cga):
        # Randomly sample # of days if unknown or anonymized
        days_cga = np.random.choice(np.arange(0, 7))
    elif reg_exps['re_d_d_slash'].findall(match_str_cga):
        days_cga = int(reg_exps['re_d_d_slash'].search(match_str_cga)
                .group(0)[0])
    elif reg_exps['re_d_d_dash'].findall(match_str_cga):
        days_cga = int(reg_exps['re_d_d_dash'].search(match_str_cga)
                .group(0)[0])
    else:
        days_cga = 0

    # Extract weeks from match
    weeks_cga = int(reg_exps['re_dd'].findall(match_str_cga)[0])

    days_ga = weeks_cga*7 + days_cga - dol
    weeks_ga_round = int(round(days_ga/7, 0))

    return days_ga, weeks_ga_round


def extract_from_ga_match(match_ga, reg_exps, verbose=0):
    # Extract matched string
    match_str_ga = match_ga.group(0)

    # Extract the days part of the gestational age
    if reg_exps['re_dd_d'].findall(match_str_ga):
        days_ga = int(reg_exps['re_dd_d'].search(match_str_ga)
                .group(0)[-1])
    # Randomly sample # of days if unknown or anonymized
    elif reg_exps['re_anon_dd_p'].findall(match_str_ga):
        days_ga = np.random.choice(np.arange(0, 7))
    elif reg_exps['re_d_d_slash'].findall(match_str_ga):
        days_ga = int(reg_exps['re_d_d_slash'].search(match_str_ga).group(0)[0])
    elif reg_exps['re_d_d_dash'].findall(match_str_ga):
        days_ga = int(reg_exps['re_d_d_dash'].search(match_str_ga).group(0)[0])
    else:
        days_ga = 0

    # Extract weeks from match
    try:
        weeks_ga = w2n.word_to_num(match_str_ga)
    except ValueError:
        weeks_ga = int(reg_exps['re_dd'].findall(match_str_ga)[0])

    # Calculate days GA
    days_ga += weeks_ga*7

    # Round weeks + days
    weeks_ga_round = int(round(days_ga/7, 0))

    return days_ga, weeks_ga_round


def extract_gest_age(s, reg_exps, verbose=0):
    match, days_ga, weeks_ga_round = None, 0 ,0
    # Reformat string to lowercase without new line characters
    s = s.replace('\n', ' ').lower()

    # See if a match can be found with the corrected gestational age regex
    match = reg_exps['re_cga'].search(s)

    if match:
        days_ga, weeks_ga_round = extract_from_cga_match(match,
                reg_exps)
    else:
        # See if a match can be found with the gestational age regex
        match = reg_exps['re_ga'].search(s)

        if match:
            days_ga, weeks_ga_round = extract_from_ga_match(
                    match, reg_exps)
        else:
            if verbose: print(f'The GA cannot be extracted from: {s}')

    return match, days_ga, weeks_ga_round

