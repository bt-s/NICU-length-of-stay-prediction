#!/usr/bin/python3

"""utils.py - Utils for extracting data of newborn patients from the
              MIMIC-III CSVs.

As part of my Master's thesis at KTH Royal Institute of Technology.
"""

__author__ = "Bas Straathof"

import pandas as pd
import os


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


def filter_on_first_admission(df):
    df_first_admin = df[['SUBJECT_ID', 'INTIME']].groupby(
            'SUBJECT_ID').min().reset_index()
    df = df[df['INTIME'].isin(df_first_admin['INTIME'])]

    return df
