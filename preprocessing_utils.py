#!/usr/bin/python3

"""preprocessing_utils.py

Various functions to preprocess variables
"""

__author__ = "Bas Straathof"

import re

import pandas as pd

from reg_exps import reg_exps


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

    # Round to integer
    v = v.astype(float).apply(lambda x: round(x))

    return v


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

    # Round to first decimal
    v = v.astype(float).apply(lambda x: round(x, 1))

    return v


def clean_vars_round_to_integer(v):
    """Round value of variables that don't need additional cleaning to integer

    Args:
        v (pd.Series): Series containing a clinical variable's values

    Returns:
        v (pd.Series): Series containing the rounded clinical variable's values
    """
    # Round values to nearest integer
    v = v.astype(float).apply(lambda x: round(x))

    return v


def clean_lab_vars_round_first_dec(v):
    """Round value of laboratory variable that don't need additional cleaning to
       first decimal

    Args:
        v (pd.Series): Series containing a laboratory variable's values

    Returns:
        v (pd.Series): Series containing the rounded laboratory variable's
                       values
    """
    # Filter out erroneous non-float values
    indices = v.astype(str).apply(lambda s:
            not re.match(reg_exps['re_lab_vals'], s))
    v.loc[indices] = None

    # Round value to nearest decimal
    v = v.astype(float).apply(lambda x: round(x, 1))

    return v


def clean_lab_vars_round_int(v):
    """Round value of laboratory variable that don't need additional cleaning
       to integer

    Args:
        v (pd.Series): Series containing a laboratory variable's values

    Returns:
        v (pd.Series): Series containing the rounded laboratory variable's
                       values
    """
    # Filter out erroneous non-float values
    indices = v.astype(str).apply(lambda s:
            not re.match(reg_exps['re_lab_vals'], s))
    v.loc[indices] = None

    # Round value to nearest integer
    v = v.astype(float).apply(lambda x: round(x) if pd.notna(x) else None)

    return v


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

    # Round to first decimal
    v = v.apply(lambda x: round(x, 1))

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

    # Round to nearest 10 grams
    v = v.apply(lambda x: round(x, 2))

    return v


cleaning_functions = {
    'BILIRUBIN_DIRECT': clean_lab_vars_round_first_dec,
    'BILIRUBIN_INDIRECT': clean_lab_vars_round_first_dec,
    'BLOOD_PRESSURE_DIASTOLIC': clean_blood_pressure_diastolic,
    'BLOOD_PRESSURE_SYSTOLIC': clean_blood_pressure_systolic,
    'CAPILLARY_REFILL_RATE': clean_capillary_refill_rate,
    'FRACTION_INSPIRED_OXYGEN': clean_vars_round_to_integer,
    'HEART_RATE': clean_vars_round_to_integer,
    'HEIGHT': clean_vars_round_to_integer,
    'OXYGEN_SATURATION': clean_lab_vars_round_int,
    'PH': clean_lab_vars_round_first_dec,
    'RESPIRATORY_RATE': clean_lab_vars_round_int,
    'TEMPERATURE': clean_temperature,
    'WEIGHT': clean_weight
}


def clean_variables(df, cleaning_functions=cleaning_functions):
    """Clean all variables in a dataframe containing all selected clinical
    event variables corresponding to a subject

    Args:
        df (pd.DataFrame): Dataframe containing clinical variables
        cleaning_functions (dict): Dictionary of variable to cleaning function
                                   mappings

    Returns:
        df (pd.DataFrame): Dataframe containing all cleaned clinical variables
                           that are not null
    """
    for var, fn in cleaning_functions.items():
        indices = (df.VARIABLE == var)

        fn = cleaning_functions[var]

        try:
            df.loc[indices, 'VALUE'] = fn(df.VALUE.loc[indices])
        except Exception as e:
            print(f'Error: {e} in {fn.__name__}')
            exit()


    return df.loc[df.VALUE.notnull()]


def remove_invalid_values(df_events, valid_ranges, value_counts):
    # Create empty dataframe
    df = pd.DataFrame(columns=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID',
        'CHARTTIME', 'ITEMID', 'VALUE', 'VALUEUOM', 'VARIABLE'])

    for var in valid_ranges.keys():
        df_var_events = df_events[df_events.VARIABLE == var]

        # Originial number of events for var
        num_events_var_orig = len(df_var_events)

        # Remove invalid values
        df_var_events = df_var_events.loc[(df_var_events.VALUE >=
            valid_ranges[var]['MIN']) & (df_var_events.VALUE <=
                valid_ranges[var]['MAX'])]
        num_invalid_values = num_events_var_orig - len(df_var_events)

        # Store all valid values
        value_counts[var]['VALUES'].extend(df_var_events.VALUE.tolist())

        # Append to df
        df = df.append(df_var_events)

        if not df_var_events.empty:
            value_counts[var]['SUBJECTS'] += 1
            value_counts[var]['INVALID_VALUES'] += num_invalid_values

    return df, value_counts

