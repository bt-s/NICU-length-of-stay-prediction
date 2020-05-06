#!/usr/bin/python3

"""mimic_readers.py - Contains class for reading data from MIMIC-III CSVs. """

__author__ = "Bas Straathof"

import pandas as pd
import os


class MimicNICUReaders(object):
    """Class to read data about newborns at the NICU from MIMIC-III tables"""
    def __init__(self, mimic_iii_path, verbose):
        self.mimic_iii_path = mimic_iii_path
        self.v_print = print if verbose else lambda *a, **k: None

    def read_admissions_table(self):
        """Read the MIMIC-III ADMISSIONS.csv table

        Returns:
            df (pd.DataFrame): Dataframe of all admissions
        """
        self.v_print('...read ADMISSIONS table...')
        df = pd.read_csv(os.path.join(self.mimic_iii_path, 'ADMISSIONS.csv'),
                dtype={'SUBJECT_ID': int, 'HADM_ID': int})

        # Make sure that the time fields are datatime
        df.ADMITTIME = pd.to_datetime(df.ADMITTIME)
        df.DISCHTIME = pd.to_datetime(df.DISCHTIME)
        df.DEATHTIME = pd.to_datetime(df.DEATHTIME)

        # Only keep relevant columns
        df = df[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME',
            'HOSPITAL_EXPIRE_FLAG', 'ADMISSION_TYPE', 'DIAGNOSIS',
            'HAS_CHARTEVENTS_DATA']]

        self.v_print(f'Total admissions identified: {df.shape[0]}')

        # Data set filtered on NICU admissions
        df = df[df.ADMISSION_TYPE == 'NEWBORN']
        self.v_print(f'Total NICU admissions identified: {df.shape[0]}\n' \
                f'Total unique NICU patients identified: ' \
                f'{df.SUBJECT_ID.nunique()}')

        # Data set filtered on newborn admissions
        df = df[df.DIAGNOSIS == 'NEWBORN']

        # Make sure that there are no duplicate SUBJECT_IDs in df
        self.v_print(f'Total newborn admissions identified: {df.shape[0]}\n' \
                f'Total unique newborn patients identified: ' \
                f'{df.SUBJECT_ID.nunique()}')

        # Only keep admissions with associated chartevents
        df = df[df.HAS_CHARTEVENTS_DATA == 1]
        self.v_print(f'Filtered newborn admissions -- with chart events: ' \
                f'{df.shape[0]}')

        return df

    def read_icustays_table(self):
        """Read the MIMIC-III ICUSTAYS.csv table

        Returns:
            df (pd.DataFrame): Dataframe of all ICU stays
        """
        self.v_print('...read ICUSTAYS table...')
        df = pd.read_csv(os.path.join(self.mimic_iii_path, 'ICUSTAYS.csv'),
            dtype={'SUBJECT_ID': int, 'HADM_ID': int, 'ICUSTAY_ID': int})

        # Make sure that the time fields are datatime
        df.INTIME = pd.to_datetime(df.INTIME)
        df.OUTTIME = pd.to_datetime(df.OUTTIME)

        # Only keep relevant columns
        df = df[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'INTIME', 'OUTTIME',
            'LOS', 'FIRST_CAREUNIT', 'LAST_CAREUNIT', 'FIRST_WARDID',
            'LAST_WARDID']]

        self.v_print(f'Total ICU stays identified: {df.shape[0]}')

        # Filter on neonatal ICU
        df= df[df.FIRST_CAREUNIT == 'NICU']
        self.v_print(f'Total NICU stays identified: {df.shape[0]}')

        # Only keep NICU stays without transfers
        df= df.loc[(df.FIRST_WARDID == df.LAST_WARDID) &
                (df.FIRST_CAREUNIT == df.LAST_CAREUNIT)]
        self.v_print(f'Filtered NICU stays -- without transfers: {df.shape[0]}')

        # Only keep the first stay
        df_first_admin = df[['SUBJECT_ID', 'INTIME']].groupby(
                'SUBJECT_ID').min().reset_index()
        df= df[df.INTIME.isin(df_first_admin.INTIME)]
        self.v_print(f'Filtered NICU stays -- first stay: {df.shape[0]}')

        # Remove admissions with undefined LOS
        df= df[df.LOS.isnull() == False]
        self.v_print(f'Filtered NICU stays -- defined LOS: {df.shape[0]}')

        # Remove admission shorter than four hours
        df= df[df.LOS >= 1/6]
        self.v_print(f'Filtered NICU stays -- longer than four hours: ' \
                f'{df.shape[0]}')

        # Create rounded LOS_HOURS variable
        df['LOS_HOURS'] = round(df.LOS * 24, 0).astype('int')

        return df

    def read_patients_table(self):
        """Read the MIMIC-III PATIENTS.csv table

        Returns:
            df (pd.DataFrame): Dataframe of all ICU stays
        """
        self.v_print('...read PATIENTS table...')
        df = pd.read_csv(os.path.join(self.mimic_iii_path, 'PATIENTS.csv'),
                dtype={'SUBJECT_ID': int})

        # Make sure that the time fields are datatime
        df.DOB = pd.to_datetime(df.DOB)
        df.DOD = pd.to_datetime(df.DOD)

        # Only keep relevant columns
        df = df[['SUBJECT_ID', 'GENDER', 'DOB', 'DOD']]

        self.v_print(f'Total patients identified: {df.shape[0]}')

        return df

    def read_noteevents_table(self):
        """Read the MIMIC-III NOTEEVENTS.csv table

        Returns:
            df (pd.DataFrame): Dataframe of all note events
        """
        self.v_print('...read NOTEEVENTS table...')
        df = pd.read_csv(os.path.join(self.mimic_iii_path, 'NOTEEVENTS.csv'))

        # Make sure that the time fields are datatime
        df.CHARTTIME = pd.to_datetime(df.CHARTTIME)

        # Only keep relevant columns
        df = df[['SUBJECT_ID', 'HADM_ID', 'CHARTDATE', 'CHARTTIME', 'CATEGORY',
            'DESCRIPTION', 'ISERROR', 'TEXT']]

        return df

    def read_labevents_table(self):
        """Read the MIMIC-III LABEVENTS.csv table

        Returns:
            df (pd.DataFrame): Dataframe of all labevents
        """
        self.v_print('...read LABEVENTS table...')
        df = pd.read_csv(os.path.join(self.mimic_iii_path, 'LABEVENTS.csv'))

        # Make sure that the time fields are datatime
        df.CHARTTIME = pd.to_datetime(df.CHARTTIME)

        # Only keep relevant columns
        df = df[['SUBJECT_ID', 'HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUE',
            'VALUENUM', 'VALUEUOM', 'FLAG']]

        # Filter df on subjects and admissions in df
        df = df[df.SUBJECT_ID.isin(df.SUBJECT_ID)]
        df = df[df.HADM_ID.isin(df.HADM_ID)]

        return df

