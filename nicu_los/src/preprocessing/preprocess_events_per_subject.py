#!/usr/bin/python3

"""preprocess_events_per_subject.py

Script to preprocess the clinical events:
    - Filter selected variables
    - Clean variables
    - Remove invalid values
"""

__author__ = "Bas Straathof"

import argparse, csv, json, os, sys

import pandas as pd
import numpy as np
import multiprocessing as mp

from tqdm import tqdm

from nicu_los.src.utils.cleaning_functions import cleaning_functions
from nicu_los.src.utils.utils import get_subject_dirs, remove_subject_dir


def parse_cl_args():
    """Parses CL arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-sp', '--subjects-path', type=str, default='data/',
            help='Path to subject directories.')
    parser.add_argument('-op', '--output-path', type=str,
            default='logs/variable_statistics.csv',
            help='Path to the output file.')
    parser.add_argument('-v', '--verbose', type=int,
            help='Level of verbosity in console output.', default=1)

    return parser.parse_args(sys.argv[1:])


def clean_variables(df, cleaning_functions):
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


def main(args):
    subjects_path, verbose = args.subjects_path, args.verbose

    with open('nicu_los/config.json') as f:
        config = json.load(f)
        vars_to_itemid = config['vars_to_itemid']
        valid_ranges = config['valid_variable_ranges']

    subject_dirs = get_subject_dirs(subjects_path)

    if verbose: print("Filtering and cleaning selected variables...")
    tot_subjects = len(subject_dirs)
    removed_subjects, tot_events, tot_events_kept = 0, 0, 0

    # Create item_id to var dictionary based on vars_to_itemid
    item_id_to_vars = {}
    for var, item_ids in vars_to_itemid.items():
        for item_id in item_ids:
            item_id_to_vars[item_id] = var

    # Create a list of variables to keep
    itemids_to_keep = list(item_id_to_vars.keys())

    # Create a pandas dataframe based on item_id_to_vars
    df_item_id  = pd.DataFrame(item_id_to_vars.items(),
            columns=['ITEMID', 'VARIABLE'])

    # Initialize variable counts dictionary
    variable_counts = {}
    for var in vars_to_itemid.keys():
        variable_counts[var] = {'VALUES': [], 'SUBJECTS': 0,
                'INVALID_VALUES': 0}

    for i, sd in enumerate(tqdm(subject_dirs)):
        # Read the events dataframe
        df_events = pd.read_csv(os.path.join(sd, 'events.csv'))

        tot_events += len(df_events)

        # Filter the dataframe on the variables that we want to keep
        df_events = pd.merge(df_events, df_item_id, how='inner', on='ITEMID')
        df_events = df_events[df_events.VALUE.notnull()]

        # Clean variables
        df_events = clean_variables(df_events)

        # Clean charttime -- we know from the format that the length should
        # always be 19
        df_events = df_events[df_events.CHARTTIME.str.len() == 19]

        # Remove invalid values
        df_events, variable_counts = remove_invalid_values(df_events,
                valid_ranges, variable_counts)

        # Sort on CHARTTIME
        df_events = df_events.sort_values(by='CHARTTIME')

        tot_events_kept += len(df_events)

        # Write df_events to CSV
        if not df_events.empty:
            df_events.to_csv(os.path.join(sd, 'events.csv'),
            index=False)
        else:
            remove_subject_dir(os.path.join(sd))
            removed_subjects += 1

    # Write results to the file
    with open(args.output_path, 'w') as wf:
        csv_header = ['VARIABLE', 'COUNT', 'SUBJECTS', 'INVALID_VALUES', 'MIN',
                'MAX', 'MEAN', 'MEDIAN']

        wf.write(','.join(csv_header) + '\n')
        csv_writer = csv.DictWriter(wf, fieldnames=csv_header,
            quoting=csv.QUOTE_MINIMAL)

        for key, val in variable_counts.items():
            results = {
                'VARIABLE': key,
                'COUNT': len(variable_counts[key]['VALUES']),
                'SUBJECTS': variable_counts[key]['SUBJECTS'],
                'INVALID_VALUES': variable_counts[key]['INVALID_VALUES'],
                'MIN': np.min(variable_counts[key]['VALUES']),
                'MAX': np.max(variable_counts[key]['VALUES']),
                'MEAN': np.mean(variable_counts[key]['VALUES']),
                'MEDIAN': np.median(variable_counts[key]['VALUES'])}

            csv_writer.writerows([results])

    if verbose:
        print(f'Of the initial {tot_subjects} subjects, ' \
                f'{tot_subjects-removed_subjects} remain that have valid ' \
                f'variables of interest associated with them.\nOf the ' \
                f'initial {tot_events} events, {tot_events_kept} remain ' \
                f'which are variables of interest.')

        total_invalid_values = 0
        for key, val in variable_counts.items():
            total_invalid_values += variable_counts[key]['INVALID_VALUES']
        print(f'The total number of invalid values is: {total_invalid_values}')


if __name__ == '__main__':
    main(parse_cl_args())

