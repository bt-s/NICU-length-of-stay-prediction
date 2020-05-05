#!/usr/bin/python3

"""cleaning_functions.py

Various functions to clean variables.
"""

__author__ = "Bas Straathof"

import csv, os, re

import pandas as pd
import numpy as np

from nicu_los.src.utils.reg_exps import reg_exps


def clean_capillary_refill_rate(v):
    """Clean the capillary refill rate (CRR) variable

    Args:
        v (pd.Series): Series containing all CRR values

    Returns:
        v (pd.Series): Series containing all cleaned CRR values
    """
    crr_map = {'Brisk': 0, 'Delayed': 1, 'Normal <3 secs': 0,
            'Abnormal >3 secs': 1}

    # Map strings to integers
    v = v.astype(str).apply(lambda x: crr_map[x] if x in crr_map else None)

    return v


def clean_blood_pressure_diastolic(v):
    """Clean the diastolic blood pressure (DBP) variable

    Args:
        v (pd.Series): Series containing all DBP values

    Returns:
        v (pd.Series): Series containing all cleaned DBP values
    """
    # Filter out values with a d/d format
    indices = v.astype(str).apply(lambda s: '/' in s)

    # Obtain diastolic bp from values with a d/d format
    v.loc[indices] = v[indices].apply(
            lambda s: re.match(reg_exps['re_d_over_d'], s).group(2))

    return v.astype(float)


def clean_blood_pressure_systolic(v):
    """Clean the systolic blood pressure (SBP) variable

    Args:
        v (pd.Series): Series containing all SBP values

    Returns:
        v (pd.Series): Series containing all cleaned SBP values
    """
    # Filter out values with a d/d format
    indices = v.astype(str).apply(lambda s: '/' in s)

    # Obtain systolic bp from values with a d/d format
    v.loc[indices] = v[indices].apply(lambda s: re.match(
        reg_exps['re_d_over_d'], s).group(1))

    return v.astype(float)


def clean_lab_vars(v):
    """Filter out erroneous non-float values from lab vars

    Args:
        v (pd.Series): Series containing a laboratory variable's values

    Returns:
        v (pd.Series): Series containing laboratory variable's cleaned values
    """
    indices = v.astype(str).apply(lambda s:
            not re.match(reg_exps['re_lab_vals'], s))
    v.loc[indices] = None

    return v.astype(float)


def clean_temperature(v):
    """Clean the temperature variable

    Args:
        v (pd.Series): Series containing all temperature values

    Returns:
        v (pd.Series): Series containing all cleaned temperature values
    """
    # Convert values to float
    v = v.astype(float)

    # Extract indices of values to be converted from Fahrenheit to Celsius
    indices = v >= 70

    # Convert Fahrenheit to Celcius
    v.loc[indices] = (v[indices] - 32.0) * 5.0 / 9.0

    return v


def clean_weight(v):
    """Clean the weight variable

    Args:
        v (pd.Series): Series containing all weight values

    Returns:
        v (pd.Series): Series containing all cleaned weight values
    """
    # Filter out erroneous non-float values
    indices = v.astype(str).apply(
            lambda x: not re.match(reg_exps['re_lab_vals'], x))
    v.loc[indices] = None

    # Convert values to float
    v = v.astype(float)

    # Sometimes the value is given in grams -- convert to kg
    indices_g = v > 100
    v.loc[indices_g] = v[indices_g].apply(lambda x: x / 1000)

    return v


cleaning_functions = {
    'BILIRUBIN_DIRECT': clean_lab_vars,
    'BILIRUBIN_INDIRECT': clean_lab_vars,
    'BLOOD_PRESSURE_DIASTOLIC': clean_blood_pressure_diastolic,
    'BLOOD_PRESSURE_SYSTOLIC': clean_blood_pressure_systolic,
    'CAPILLARY_REFILL_RATE': clean_capillary_refill_rate,
    'OXYGEN_SATURATION': clean_lab_vars,
    'PH': clean_lab_vars,
    'RESPIRATORY_RATE': clean_lab_vars,
    'TEMPERATURE': clean_temperature,
    'WEIGHT': clean_weight
}

