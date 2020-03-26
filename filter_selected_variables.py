from tqdm import tqdm
from sys import argv

import os
import csv
import argparse
import shutil

import pandas as pd

vars_to_itemid = {
    "Bilirubin": [
        50885, 51464, 51465, 1538, 50883, 50884, 848, 225690,
        803, 3765, 225651, 1527, 50838, 51028, 51049, 51465
    ],
    "Blood Pressure -- Diastolic": [
        8368, 8441, 220180, 220051, 8502, 225310, 8555, 8440,
        8503, 8504, 8506, 8507, 224643, 227242
    ],
    "Blood Pressure -- Mean": [
       52, 456, 220181, 220052, 3312, 225312, 224, 6702, 224322,
       3314, 3316, 3322, 3320
    ],
    "Blood Pressure -- Systolic": [
       51, 455, 220179, 220050, 3313, 225309, 6701, 3315, 442,
       3317, 3323, 3321, 224167, 227243,
    ],
    "Capillary Refill Rate": [
       3348, 115, 8377
    ],
    "Fraction Inspired Oxygen": [
       3420, 223835, 3422, 189, 727
    ],
    "Glucose": [
       50931, 807, 811, 1529, 225664, 50809, 220621, 51478,
       226537, 3745, 3744,
    ],
    "Heart Rate": [
       211, 220045
    ],
    "Height": [
       226707, 226730, 1394, 4188, 4187 # Check 4188 and 4187
    ],
    "Oxygen Saturation": [
       646, 220277, 834, 50817, 220227, 8498
    ],
    "pH": [
       50820, 780, 1126, 223830, 51491, 220734, 860, 4753,
       220274, 3839, 4202, 1673, 50831, 51094 # Check 220734
    ],
    "Respiratory Rate": [
       678, 220210, 3603, 224689, 615, 224690, 614, 651,
       224422
    ],
    "Temperature": [
       678, 677, 3655, 223761, 676, 679, 223762, 3654
    ],
    "Weight": [
        3580, 3581, 3582, 763, 224639, 226531, 226512,
        3723, 4183, 3693,
        # Birth weight: 3723, 4183
        # Previous weight: 580, 581, 3583
        # Weight change: 3692, 733
        # Feeding weight: 226846
    ],
}


def parse_cl_args():
    """Parses CL arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-sp', '--subjects-path', type=str, default='data/',
            help='Path to subject directories.')
    parser.add_argument('-v', '--verbose', type=int,
            help='Level of verbosity in console output.', default=1)

    return parser.parse_args(argv[1:])


def main(args):
    verbose, subjects_path = args.verbose, args.subjects_path
    subject_directories = os.listdir(subjects_path)
    subject_directories = set(filter(lambda x: str.isdigit(x),
        subject_directories))
    tot_subjects = len(subject_directories)
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
    for item_id in itemids_to_keep:
        variable_counts[item_id] = 0

    for i, subject_dir in enumerate(tqdm(subject_directories)):
        # Read the events dataframe
        df_events = pd.read_csv(os.path.join(subjects_path, str(subject_dir),
            'events.csv'))

        tot_events += len(df_events)

        # Filter the dataframe
        df_events = pd.merge(df_events, df_item_id, how='inner', on='ITEMID')
        for k, v in df_events.ITEMID.value_counts().to_dict().items():
            variable_counts[k] += v
        tot_events_kept += len(df_events)

        # Write df_events to events.csv if not empty, remove otherwise
        if not df_events.empty:
            # Sort on CHARTTIME
            df_events = df_events.sort_values(by='CHARTTIME')
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
                f'{tot_events_kept} remain which are ' \
                f'variables of interest.')

        for k, v in variable_counts.items():
            print(f'Variable {k} - {item_id_to_vars[k]} occurs, {v} ' \
                    'times')


if __name__ == '__main__':
    main(parse_cl_args())

