#!/usr/bin/python3

"""csv_readers.py - Functions for reading data from MIMIC-III CSVs.

As part of my Master's thesis at KTH Royal Institute of Technology.
"""

__author__ = "Bas Straathof"

import pandas as pd
import os


def read_admissions_table(mimic_iii_path):
    """Read the MIMIC-III ADMISSIONS.csv table

    Args:
        mimic_iii_path (str): Path to the MIMIC-III CSVs

    Returns:
       df (pd.DataFrame): Dataframe of all admissions
    """
    df = pd.read_csv(os.path.join(mimic_iii_path, 'ADMISSIONS.csv'),
            dtype={'SUBJECT_ID': int, 'HADM_ID': int})

    # Make sure that the time fields are datatime
    df.ADMITTIME = pd.to_datetime(df.ADMITTIME)
    df.DISCHTIME = pd.to_datetime(df.DISCHTIME)
    df.DEATHTIME = pd.to_datetime(df.DEATHTIME)

    # Only keep relevant columns
    df = df[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME',
        'HOSPITAL_EXPIRE_FLAG', 'ADMISSION_TYPE', 'DIAGNOSIS',
        'HAS_CHARTEVENTS_DATA']]

    return df


def read_icustays_table(mimic_iii_path):
    """Read the MIMIC-III ICUSTAYS.csv table

    Args:
        mimic_iii_path (str): Path to the MIMIC-III CSVs

    Returns:
       df (pd.DataFrame): Dataframe of all ICU stays
    """
    df = pd.read_csv(os.path.join(mimic_iii_path, 'ICUSTAYS.csv'),
        dtype={'SUBJECT_ID': int, 'HADM_ID': int, 'ICUSTAY_ID': int})

    # Make sure that the time fields are datatime
    df.INTIME = pd.to_datetime(df.INTIME)
    df.OUTTIME = pd.to_datetime(df.OUTTIME)

    # Only keep relevant columns
    df = df[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'INTIME', 'OUTTIME', 'LOS',
        'FIRST_CAREUNIT', 'LAST_CAREUNIT', 'FIRST_WARDID', 'LAST_WARDID']]

    return df


def read_patients_table(mimic_iii_path):
    """Read the MIMIC-III PATIENTS.csv table

    Args:
        mimic_iii_path (str): Path to the MIMIC-III CSVs

    Returns:
       df (pd.DataFrame): Dataframe of all ICU stays
    """
    df = pd.read_csv(os.path.join(mimic_iii_path, 'PATIENTS.csv'),
            dtype={'SUBJECT_ID': int})

    # Make sure that the time fields are datatime
    df.DOB = pd.to_datetime(df.DOB)
    df.DOD = pd.to_datetime(df.DOD)

    # Only keep relevant columns
    df = df[['SUBJECT_ID', 'GENDER', 'DOB', 'DOD']]

    return df


def read_noteevents_table(mimic_iii_path):
    df = pd.read_csv(os.path.join(mimic_iii_path, 'NOTEEVENTS.csv'))

    # Make sure that the time fields are datatime
    df.CHARTTIME = pd.to_datetime(df.CHARTTIME)

    # Only keep relevant columns
    df = df[['SUBJECT_ID', 'HADM_ID', 'CHARTDATE', 'CHARTTIME', 'CATEGORY',
        'DESCRIPTION', 'ISERROR', 'TEXT']]

    return df


def read_labevents_table(mimic_iii_path):
    df = pd.read_csv(mimic_iii_path + 'LABEVENTS.csv')

    # Make sure that the time fields are datatime
    df.CHARTTIME = pd.to_datetime(df.CHARTTIME)

    # Only keep relevant columns
    df = df[['SUBJECT_ID', 'HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUE',
        'VALUENUM', 'VALUEUOM', 'FLAG']]

    return df

