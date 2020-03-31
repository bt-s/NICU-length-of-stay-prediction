import pandas as pd
import re
from reg_exps import reg_exps

from tqdm import tqdm
from sys import argv
import os
import argparse
import shutil
import datetime


def parse_cl_args():
    """Parses CL arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-sp', '--subjects-path', type=str, default='data/',
            help='Path to subject directories.')
    parser.add_argument('-v', '--verbose', type=int,
            help='Level of verbosity in console output.', default=1)

    return parser.parse_args(argv[1:])


def clean_capillary_refill_rate(df):
    crr_map = {'Brisk': 0, 'Delayed': 1, 'Normal <3 secs': 0,
            'Abnormal >3 secs': 1}

    # Read values as string
    v = df.VALUE.astype(str)

    # Map strings to integers
    v = v.apply(lambda x: crr_map[x] if x in crr_map else None)

    return pd.to_numeric(v, errors='coerce')


def clean_diastolic_blood_pressure(df):
    # Read values as string
    v = df.VALUE.astype(str)

    # Filter out values with a d/d format
    indices = v.apply(lambda s: '/' in s)

    # Obtain diastolic bp from values with a d/d format
    v.loc[indices] = v[indices].apply(
            lambda s: re.match(reg_exps['re_d_over_d'], s).group(2))

    # Convert to float
    v = v.astype(float)

    # Round to first decimal
    v = v.apply(lambda x: round(x, 1))

    return pd.to_numeric(v, errors='coerce')


def clean_vars_round_to_integer(df):
    # Read values as float
    v = df.VALUE.astype(float)

    # Round values to nearest integer
    v = v.apply(lambda x: round(x))

    return pd.to_numeric(v, errors='coerce')


def clean_lab_vars_round_first_dec(df):
    # Read values as string
    v = df.VALUE.astype(str)

    # Filter out erroneous non-float values
    indices = v.apply(lambda s: not re.match(reg_exps['re_lab_vals'], s))
    v.loc[indices] = None

    # Convert values to float
    v = v.astype(float)

    # Round value to nearest decimal
    v = v.apply(lambda x: round(x, 1))

    return pd.to_numeric(v, errors='coerce')


def clean_lab_vars_round_int(df):
    # Read values as string
    v = df.VALUE.astype(str)

    # Filter out erroneous non-float values
    indices = v.apply(lambda s: not re.match(reg_exps['re_lab_vals'], s))
    v.loc[indices] = None

    # Convert values to float
    v = v.astype(float)

    # Round value to nearest integer
    v = v.apply(lambda x: round(x) if pd.notna(x) else None)

    return pd.to_numeric(v, errors='coerce')


def clean_systolic_blood_pressure(df):
    # Read values as string
    v = df.VALUE.astype(str)

    # Filter out values with a d/d format
    indices = v.apply(lambda s: '/' in s)

    # Obtain diastolic bp from values with a d/d format
    v.loc[indices] = v[indices].apply(lambda s: re.match(
        reg_exps['re_d_over_d'], s).group(1))

    # Convert values to float
    v = v.astype(float)

    # Round to first decimal
    v = v.apply(lambda x: round(x, 1))

    return pd.to_numeric(v, errors='coerce')


def clean_temperature(df):
    temp_map = {3655: 'C', 3654: 'F' }

    # Read values as string
    v = df.VALUE.astype(float)

    # Extract indices of values to be converted
    indices = df.ITEMID.apply(lambda x: temp_map[x] == 'F') | (v >= 75)

    # Convert Fahrenheit to Celcius
    v.loc[indices] = (v[indices] - 32.0) * 5.0 / 9.0

    # Round to first decimal
    v = v.apply(lambda x: round(x, 1))

    return pd.to_numeric(v, errors='coerce')


def clean_weight(df):
    # Some rows contain errors
    v = df.VALUE.astype(str)

    # Filter out erroneous non-float values
    indices = v.apply(lambda x: not re.match(reg_exps['re_lab_vals'], x))
    v.loc[indices] = None

    # Convert values to float
    v = v.astype(float)

    # Sometimes the value is given in grams -- convert to kg
    indices_g = v > 100
    v.loc[indices_g] = v[indices_g].apply(lambda x: x / 1000)

    # Round to nearest 100 grams
    v = v.apply(lambda x: round(x, 1))

    return pd.to_numeric(v, errors='coerce')


cleaning_functions = {
    'Bilirubin -- Direct': clean_lab_vars_round_first_dec,
    'Bilirubin -- Indirect': clean_lab_vars_round_first_dec,
    'Blood Pressure -- Diastolic': clean_diastolic_blood_pressure,
    'Blood Pressure -- Systolic': clean_systolic_blood_pressure,
    'Capillary Refill Rate': clean_capillary_refill_rate,
    'Fraction Inspired Oxygen': clean_vars_round_to_integer,
    'Heart Rate': clean_vars_round_to_integer,
    'Height': clean_vars_round_to_integer,
    'Oxygen Saturation': clean_lab_vars_round_int,
    'pH': clean_lab_vars_round_first_dec,
    'Respiratory Rate': clean_lab_vars_round_int,
    'Temperature': clean_temperature,
    'Weight': clean_weight}


def clean_variables(df):
    global cleaning_functions

    for var, fn in cleaning_functions.items():
        indices = (df.VARIABLE == var)

        fn = cleaning_functions[var]

        try:
            df.loc[indices, 'VALUE'] = fn(df.loc[indices])
        except Exception as e:
            print(f'Error: {e} in {fn.__name__}')
            exit()

    return df.loc[df.VALUE.notnull()]


def main(args):
    verbose, subjects_path = args.verbose, args.subjects_path
    subject_directories = os.listdir(subjects_path)
    subject_directories = set(filter(lambda x: str.isdigit(x),
        subject_directories))
    tot_subjects = len(subject_directories)
    removed_subjects, tot_events, tot_events_kept = 0, 0, 0

    for i, subject_dir in enumerate(tqdm(subject_directories)):
        # Read the events dataframe
        df_events = pd.read_csv(os.path.join(subjects_path, str(subject_dir),
            'events.csv'))

        tot_events += len(df_events)

        # Clean variables
        df_events = clean_variables(df_events)

        # Clean charttime -- we know from the format that the length should always be 19
        df_events = df_events[df_events.CHARTTIME.str.len() == 19]

        tot_events_kept += len(df_events)

        # Write df_events to events.csv if not empty, remove otherwise
        if not df_events.empty:
            df_events.to_csv(os.path.join(subjects_path, str(subject_dir),
                'events.csv'), index=False)
        else:
            try:
                shutil.rmtree(os.path.join(subjects_path, str(subject_dir)))
                removed_subjects += 1
            except OSError as e:
                print (f'Error: {e.filename} - {e.strerror}.')

    if verbose:
        print(f'Of the initial {tot_subjects} subjects, ' \
                f'{tot_subjects-removed_subjects} remain that have ' \
                f'variables of interest associated with them.\n' \
                f'Of the initial {tot_events} events, ' \
                f'{tot_events_kept} remain which are cleaned ' \
                f'variables of interest.')


if __name__ == '__main__':
    main(parse_cl_args())

