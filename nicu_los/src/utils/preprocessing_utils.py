#!/usr/bin/python3

"""preprocessing_utils.py

Various functions to preprocess variables
"""

__author__ = "Bas Straathof"

import csv, os, re

import pandas as pd
import numpy as np

from tqdm import tqdm
from word2number import w2n
from datetime import datetime, timedelta
from ..utils.reg_exps import reg_exps
from ..utils.utils import remove_subject_dir


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
    """Remove clinical events with a value outside the valid range

    Args:
        df_events (pd.DataFrame): Containing the clinical events associated
                                  with a subject
        valid_ranges (dict): Mapping from variables to their valid min and max
                             values
        value_counts (dict): Dictionary storing values for statistics

    Returns:
        df (pd.DataFrame): Only containing valid clinical event values
        value_counts (dict): Appended dictionary for statistics
    """
    # Create empty dataframe
    df = pd.DataFrame(columns=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID',
        'CHARTTIME', 'ITEMID', 'VALUE', 'VALUEUOM', 'VARIABLE'])

    for var in valid_ranges.keys():
        df_var_events = df_events.loc[df_events.VARIABLE == var]
        df_var_events.VALUE = df_var_events.VALUE.astype(float)

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


def filter_on_newborns(df):
    """Filter df on patients zero days of age

    Args:
        df (pd.DataFrame): Admissions dataframe

    Returns:
        df (pd.DataFrame): Filtered dataframe
    """
    df['AGE'] = (df['ADMITTIME'] - df['DOB']).dt.days
    df = df.loc[df['AGE'] == 0]
    df = df.drop(['AGE'], axis=1)

    return df


def extract_from_cga_match(match_str_cga, reg_exps):
    """Extract the gestational age (GA) in weeks and days from a
    match of the corrected GA (CGA)

    Args:
        match_str_cga (str): String containing the CGA match
        reg_exps (dict): All regular expressions

    Returns:
        days_ga (int): GA in days
        weeks_ga_round (int): Rounded GA in weeks
    """
    match_str = reg_exps['re_splitter'].split(match_str_cga)
    match_str_dol, match_str_cga = match_str[0], match_str[2]

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


def extract_from_ga_match(match_str_ga, reg_exps):
    """Extract the gestational age (GA) in weeks and days from a
    direct match of the GA

    Args:
        match_str_ga (str): String containing the GA match
        reg_exps (dict): All regular expressions

    Returns:
        days_ga (int): GA in days
        weeks_ga_round (int): Rounded GA in weeks
    """
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


def extract_gest_age_from_note(s, reg_exps, verbose=False):
    """Extract the gestational age (GA) in weeks and days from a
    clinical note

    Args:
        s (str): String containing the test of the clinical note
        reg_exps (dict): All regular expressions
        verbose (bool): Verbosity flag

    Returns:
        match_str (str): String of the match containing the (C)GA
        max_days_ga (int): Maximum extracted GA in days
        max_weeks_ga_round (int): Maximum extracted ounded GA in weeks
    """
    # We want to find the maximum reported value in the clinical note
    match_str, max_days_ga, max_weeks_ga_round = None, 0, 0

    # Reformat string to lowercase without new line characters
    s = s.replace('\n', ' ').lower()

    # Filter out false string that occurs in many notes
    s = re.sub(reg_exps['re_false'], '', s)

    # See if a match can be found with the corrected gestational age regex
    # Assumption: if mentioned, the CGA is only mentioned once
    match = reg_exps['re_cga'].search(s)

    if match:
        # Extract string from match
        match_str = match.group(0)

        if not re.match(reg_exps['re_not_allowed'], match_str):
            days_ga, weeks_ga_round = extract_from_cga_match(match_str,
                    reg_exps)
            if (23 < weeks_ga_round < 43):
                max_weeks_ga_round = weeks_ga_round
                max_days_ga = days_ga
            else:
                match_str = None
        else:
            match_str = None
    else:
        # See if matches can be found with the gestational age regex
        matches = reg_exps['re_ga'].findall(s)

        if len(matches) != 0:
            # Extract the match with the highest gestational age
            for m in range(len(matches)):
                if not re.match(reg_exps['re_not_allowed'], matches[m][0]):
                    days_ga, weeks_ga_round = extract_from_ga_match(
                            matches[m][0], reg_exps)
                    if ((weeks_ga_round > max_weeks_ga_round) and
                            (23 < weeks_ga_round < 43)):
                        max_weeks_ga_round = weeks_ga_round
                        max_days_ga = days_ga
                        match_str = matches[m][0]
        else:
            if verbose: print(f'The GA cannot be extracted from: {s}')

    return match_str, max_days_ga, max_weeks_ga_round


def transfer_filter(s, reg_exps, verbose=False):
    """Check if note describes stay with a capacity-related transfer

    Args:
        s (str): Text of the note
        reg_exps (dict): All regular expressions
        verbose (bool): Verbosity flag

    Returns:
        match (re.Match/None): Matched regular expression
    """
    # Default: no match is found
    match = None

    # Reformat string to lowercase without new line characters
    s = s.replace('\n', ' ').lower()

    # See if a match can be found with the unpredictable transfer filter regex
    match = reg_exps['re_trans_filter'].search(s)

    return match


def process_notes(df_notes, reg_exps, verbose=True):
    """ Process notes: extract gestational age and filter transfers

    Args:
        df_notes (pd.DataFrame): Dataframe containing the notes
        reg_exps (dict): Regular expressions
        verbose (bool): Verbosity flag

    Returns:
        gest_age_set (set): Set of gestational ages
        cap_trans_set (set): Set of capacity-related transfers
    """
    df_ga = pd.DataFrame(columns=['SUBJECT_ID', 'GA_MATCH', 'GA_DAYS',
        'GA_WEEKS_ROUND'])
    gest_age_set, cap_trans_set = set(), set()
    for ix, row in tqdm(df_notes.iterrows(), total=df_notes.shape[0]):
        m_ga, m_ct = None, None # Default is that no match will be found

        # Look for GA
        if row.SUBJECT_ID not in gest_age_set:
            m_ga, d, w = extract_gest_age_from_note(row.TEXT, reg_exps)
            if m_ga:
                gest_age_set.add(row.SUBJECT_ID)
                df_ga.loc[ix] = [row.SUBJECT_ID] + [m_ga] + [d] + [w]

        # Look for capacity-related transfers
        if row.SUBJECT_ID not in cap_trans_set:
            m_ct = transfer_filter(row.TEXT, reg_exps)
            if m_ct:
                cap_trans_set.add(row.SUBJECT_ID)

    if verbose:
        print(f'Total GAs identified: {len(gest_age_set)}\nTotal ' \
                f'capacity-related admissions identified: {len(cap_trans_set)}')

    return df_ga, cap_trans_set


def los_hours_to_target(hours):
    """Convert LOS in hours to targets

    The targets exist of ten buckets:
        0: less than 1 day; 1: 1 day; 2: 2 day; 3: 3 day; 4: 4 day;
        5: 5 day; 6: 6 day; 7: 7 day; 8: 8-13 days; 9: more than 14 days

    Args:
        hours (int): LOS in hours

    Return:
        target (int): The respective target
    """
    if hours < 24:
        target = 0
    elif 24 <= hours < 48:
        target = 1
    elif 48 <= hours < 72:
        target = 2
    elif 72 <= hours < 96:
        target = 3
    elif 96 <= hours < 120:
        target = 4
    elif 120 <= hours < 144:
        target = 5
    elif 144 <= hours < 168:
        target = 6
    elif 168 <= hours < 192:
        target = 7
    elif 192 <= hours < 336:
        target = 8
    elif 336 <= hours:
        target = 9

    return target


def get_first_valid_value_from_ts(ts, variable):
    """Get the first valid value of a variable in a time series

    Args:
        ts (pd.DataFrame): Timeseries dataframe
        variable (str): Name of the variable

    Returns:
        value (float): First valid value of variable
    """
    # Assume no value exists
    value = np.nan
    if variable in ts:
        # Find the indices of the rows where variable has a value in ts
        indices = ts[variable].notnull()
        if indices.any():
            index = indices.to_list().index(True)
            value = ts[variable].iloc[index]

    return value


def read_and_split_table_by_subject(mimic_iii_path, table_name, output_path,
    subjects_to_keep=None, verbose=True, n=0):
    """Read and split the MIMIC-III event table by subject

    Args:
        mimic_iii_path (str): Path to the MIMIC-III files
        table_name (str): Name of the table to read and split
        output_path (str): Where to write the extracted data
        subjects_to_keep (list): List of subjects IDs of subjects to keep
        verbose (bool): Verbosity flag
        n (int): Process ID
    """
    # Allow the table name to be passed both with lower- and uppercase letters
    table_name = table_name.upper()

    if table_name not in ['CHARTEVENTS', 'LABEVENTS', 'NOTEEVENTS']:
        raise ValueError("Table name must be one of: 'chartevents', " +
             "'labevents', 'noteevents'")
    else:
        rows_per_table = {'CHARTEVENTS': 330712484, 'LABEVENTS': 27854056,
                'NOTEEVENTS': 2083180}
        tot_nb_rows = rows_per_table[table_name]

    # Create a header for the new CSV files to be created
    if table_name == 'NOTEEVENTS':
        csv_header = ['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'CATEGORY',
            'DESCRIPTION', 'ISERROR', 'TEXT']
    else:
        csv_header = ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'CHARTTIME',
                'ITEMID', 'VALUE', 'VALUEUOM']

    def write_row_to_file():
        # Define a filename for file holding the data of current_subject_id
        subject_f = os.path.join(output_path, str(current_subject_id))

        # Create the output directory
        if not os.path.exists(subject_f):
            os.makedirs(subject_f)

        if table_name == 'NOTEEVENTS':
            subject_data_f = os.path.join(subject_f, 'notes.csv')
        else:
            subject_data_f = os.path.join(subject_f, 'events.csv')

        # Create the file and give it its header if it doesn't exist yet
        if not os.path.exists(subject_data_f) or \
                not os.path.isfile(subject_data_f):
            f = open(subject_data_f, 'w')
            f.write(','.join(csv_header) + '\n')
            f.close()

        # Write current row to the file
        with open(subject_data_f, 'a') as wf:
            csv_writer = csv.DictWriter(wf, fieldnames=csv_header,
                quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerows(objects_to_write)

    # Create variables to store the objects to write and the current subject ID
    objects_to_write, current_subject_id = [], ''

    with open(os.path.join(mimic_iii_path, table_name + '.csv')) as table:
        # Create an iterative CSV reader that outputs a row to a dictionary
        csv_reader = csv.DictReader(table)

        rows_written = 0
        if verbose: print(f'Read {table_name.upper()}.csv...')
        for i, row in enumerate(tqdm(csv_reader,
            total = rows_per_table[table_name], position=n)):
            if subjects_to_keep and (int(row['SUBJECT_ID']) not in
                    subjects_to_keep):
                continue

            if table_name == 'NOTEEVENTS':
                row_output = {'SUBJECT_ID': row['SUBJECT_ID'],
                        'HADM_ID': row['HADM_ID'],
                        'CHARTTIME': row['CHARTTIME'],
                        'CATEGORY': row['CATEGORY'],
                        'DESCRIPTION': row['DESCRIPTION'],
                        'ISERROR': row['ISERROR'],
                        'TEXT': row['TEXT']}
            else:
                row_output = {'SUBJECT_ID': row['SUBJECT_ID'],
                        'HADM_ID': row['HADM_ID'],
                        'ICUSTAY_ID': '' if 'ICUSTAY_ID' not in row else \
                                row['ICUSTAY_ID'],
                        'CHARTTIME': row['CHARTTIME'],
                        'ITEMID': row['ITEMID'],
                        'VALUE': row['VALUE'],
                        'VALUEUOM': row['VALUEUOM']}

            # Only write row to file if current_subject_id changes
            if current_subject_id != '' and \
                    current_subject_id != row['SUBJECT_ID']:
                write_row_to_file()
                objects_to_write = []

            objects_to_write.append(row_output)
            current_subject_id = row['SUBJECT_ID']

            # Increment rows_written
            rows_written += 1

        if i == tot_nb_rows:
            write_row_to_file()
            objects_to_write = []

        if verbose:
            print(f'Processed {i+1}/{tot_nb_rows} rows in '
                    f'{table_name.lower()}.csv\nIdentified '
                    f'{rows_written} events in {table_name.lower()}.')


def validate_events_and_notes_per_subject(df_stay, df_events, df_notes, stats):
    """Validate clinical events and notes per subject

    Args:
        df_stay (pd.DataFrame): Containing the ICU stay information associated
                                with the subject
        df_events (pd.DataFrame): Containing the clinical events associated
                                  with the subject
        df_notes (pd.DataFrame): Containing the clinical notes associated with
                                 the subject
        stats (dict): Containing stats about the validation procedure

    Returns:
        df_events (pd.DataFrame): Validated clinical events
        df_notes (pd.DataFrame): Validated clinical notes
    """
    tot_events, tot_notes = len(df_events), len(df_notes)

    # Make sure that the charttime field is datatime
    df_events.CHARTTIME = pd.to_datetime(df_events.CHARTTIME)
    df_notes.CHARTTIME = pd.to_datetime(df_notes.CHARTTIME)

    # Obtain the HADM_ID and ICUSTAY_ID from df_stay
    hadm_id = df_stay.HADM_ID.tolist()[0]
    icustay_id = df_stay.ICUSTAY_ID.tolist()[0]

    # Obtain INTIME and OUTTIME from df_stay and round down to hour
    date_format = "%Y-%m-%d %H:%M:%S"
    intime = df_stay.INTIME.tolist()[0]
    intime = datetime.strptime(intime, date_format)
    intime = intime.replace(microsecond=0, second=0, minute=0)
    outtime = df_stay.OUTTIME.tolist()[0]
    outtime = datetime.strptime(outtime, date_format)
    outtime = outtime.replace(microsecond=0, second=0, minute=0)

    # Drop events without a charttime
    df_events = df_events.loc[df_events.CHARTTIME.notna()]
    no_charttime = tot_events - len(df_events)

    # Drop events without a value
    df_events = df_events.loc[df_events.VALUE.notna()]
    no_value = tot_events - len(df_events) - no_charttime

    # Make sure that if VALUEOM is not present that it is an empty string
    df_events.VALUEUOM= df_events.VALUEUOM.fillna('').astype(str)

    # Drop events that fall outside the intime-outtime interval
    mask = (intime <= df_events.CHARTTIME) & (df_events.CHARTTIME < outtime)
    df_events = df_events.loc[mask]
    incorrect_charttime = tot_events - len(df_events) - no_value - no_charttime

    # Drop events for which both HADM_ID and ICUSTAY_ID are empty
    df_events = df_events.dropna(how='all', subset=['HADM_ID',
        'ICUSTAY_ID'])
    no_hadm_and_icustay = tot_events - len(df_events) - no_value \
            - no_charttime - incorrect_charttime

    # Keep events without HADM_ID but with correct ICUSTAY_ID as well as
    # events without ICUSTAY_ID but with correct HADM_ID
    df_events = df_events.fillna(value={'HADM_ID': hadm_id,
        'ICUSTAY_ID': icustay_id})

    # Drop events with wrong HADM_ID
    df_events = df_events.loc[df_events.HADM_ID == hadm_id]
    wrong_hadm = tot_events - len(df_events) - no_value - no_charttime \
            - incorrect_charttime - no_hadm_and_icustay

    # Drop events with wrong ICUSTAY_ID
    df_events = df_events.loc[df_events.ICUSTAY_ID == icustay_id]
    wrong_icustay = tot_events - len(df_events) - no_value - no_charttime \
            - incorrect_charttime - no_hadm_and_icustay - wrong_hadm

    # Drop notes without a charttime
    df_notes = df_notes.loc[df_notes.CHARTTIME.notna()]

    # Drop notes that fall outside the intime-outtime interval
    mask = (intime <= df_notes.CHARTTIME) & (df_notes.CHARTTIME < outtime)
    df_notes = df_notes.loc[mask]
    incorrect_charttime = tot_notes - len(df_notes)

    # Drop notes without a text
    df_notes = df_notes[(df_notes.TEXT.notna())]
    no_text = tot_notes - len(df_notes) - incorrect_charttime

    # Drop notes for which HADM_ID
    df_notes = df_notes.loc[df_notes.HADM_ID.notna()]

    # Drop notes with wrong HADM_ID
    df_notes = df_notes.loc[df_notes.HADM_ID == hadm_id]
    incorrect_hadm_id = tot_notes - len(df_notes) - no_text - \
            incorrect_charttime

    stats['events_tot_nb_events'] += tot_events
    stats['events_no_value'] += no_value
    stats['events_no_charttime'] += no_charttime
    stats['events_incorrect_charttime'] += incorrect_charttime
    stats['events_no_hadm_id_and_icustay_id'] += no_hadm_and_icustay
    stats['events_incorrect_hadm_id'] += wrong_hadm
    stats['events_incorrect_icustay_id'] += wrong_icustay
    stats['events_final_nb_events'] += len(df_events)
    stats['notes_tot_nb_notes'] += tot_notes
    stats['notes_no_text'] += no_text
    stats['notes_incorrect_charttime'] += incorrect_charttime
    stats['notes_incorrect_hadm_id'] += incorrect_hadm_id
    stats['notes_final_nb_notes'] += len(df_notes)

    return df_events, df_notes, stats


def validate_events_and_notes(subject_directories,
        stats_output_directory="logs/validation_statistics.csv"):
    """Validate clinical events and notes

    Args:
        subject_directories (List[str]): Containing the paths to the subject directories
        stats_output_directoryyy (str): Path to the validation statistics output CSV
    """
    tot_subjects = len(subject_directories)
    removed_subjects = 0
    stats = {'events_tot_nb_events': 0, 'events_no_value': 0,
            'events_no_charttime': 0, 'events_incorrect_charttime': 0,
            'events_no_hadm_id_and_icustay_id': 0,
            'events_incorrect_hadm_id': 0, 'events_incorrect_icustay_id': 0,
            'events_final_nb_events': 0, 'notes_tot_nb_notes': 0,
            'notes_no_text': 0, 'notes_incorrect_charttime': 0,
            'notes_incorrect_hadm_id': 0, 'notes_final_nb_notes': 0}

    for i, sd in enumerate(tqdm(subject_directories)):
        # Read the stay data for current subject
        df_stay = pd.read_csv(os.path.join(sd, 'stay.csv'))

        # Read the events for the current subject
        df_events = pd.read_csv(os.path.join(sd, 'events.csv'))

        # Read the notes for the current subject
        df_notes = pd.read_csv(os.path.join(sd, 'notes.csv'))

        # Validate events and notes
        df_events, df_notes, stats = validate_events_and_notes_per_subject(
                df_stay, df_events, df_notes, stats)

        if (not df_events.empty) and (not df_notes.empty) :
            # Write df_events to events.csv
            df_events.to_csv(os.path.join(sd, 'events.csv'), index=False)

            # Write df_notes to notes.csv
            df_notes.to_csv(os.path.join(sd, 'notes.csv'), index=False)
        else:
            remove_subject_dir(sd)
            removed_subjects += 1

    # Write results to the file
    with open(stats_output_directory, 'w') as wf:
        writer = csv.writer(wf)
        for key, value in stats.items():
            writer.writerow([key, value])

    for k, v in stats.items():
        print(k+':', v)
    print(f'From the initial {tot_subjects} subjects, ' \
            f'{tot_subjects-removed_subjects} remain that have ' \
            'events associated with them.')


