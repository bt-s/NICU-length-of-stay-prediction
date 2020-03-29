from tqdm import tqdm
from sys import argv

import os
import csv
import argparse
import shutil

import pandas as pd

vars_to_itemid = {
    "Bilirubin -- Direct": [
        50883, 803,
        # No occurences
        # 225651, 1527
    ],
    "Bilirubin -- Indirect": [
        50884, 3765
    ],
    # Implicit in the combination of direct and indirect
    #"Bilirubin -- Total": [
        #50885, 1538
        # Ambiguous or no occurrences
        # 51464, 51465, 51028, 848, 225690, 50838, 51049
    #],
    "Blood Pressure -- Diastolic": [
        8502, 8503, 8504, 8506, 8507,
        # No occurrences
        # 8368, 8441, 220180, 220051, 225310, 8555, 8440, 224643, 227242
    ],
    # Implicit in the combination of diastolic and systolic
    #"Blood Pressure -- Mean": [
    #3312, 3314, 3316, 3322, 3320
    # No occurrences
    # 52, 456, 220181, 220052, 225312, 224, 6702, 224322
    #],
    "Blood Pressure -- Systolic": [
        3313, 3315, 3317, 3323, 3321
        # No occurrences
        # 51, 455, 220179, 220050, 225309, 6701, 442, 224167, 227243
    ],
    "Capillary Refill Rate": [
        3348
        # No occurrences
        # 115, 8377
    ],
    "Fraction Inspired Oxygen": [
        3420, 3422
        # No occurrences
        # 223835, 189, 727
    ],
    #"Glucose": [ # Only ~2000 data points -- insufficient
    #50931, 50809, 51478, 3745, 3744
    # No occurrences
    # 807, 811, 1529, 225664, 220621, 226537
    #],
    "Heart Rate": [
        211
        # No occurrences
        # 220045
    ],
    "Height": [
        4188
        # Wrong unit
        # 4187
        # No occurrences
        # 226707, 226730, 1394,
    ],
    "Oxygen Saturation": [
        834, 50817, 8498
        # No occurrences
        # 646, 220227, 220227
    ],
    "pH": [
        50820, 51491, 860, 4753, 3839, 4202, 1673, 50831, 51094
        # No occurrences
        # 780, 1126, 223830, 220734, 220274
    ],
    "Respiratory Rate": [
        3603,
        # No occurrences
        # 220210, 224689, 615, 224690, 614, 651, 224422
    ],
    "Temperature": [
        3655, 3654
        # No occurrences
        # 676, 677, 678, 679, 223761, 223762
    ],
    "Weight": [
        3580, 3693,
        3723, 4183 # birth weight

        # No occurrences
        # 763, 224639, 226531, 226512

        # Wrong unit
        # 3581, 3582

        # Previous weight
        # 580, 581, 3583

        # Weight change
        # 3692, 733

        # Feeding weight
        # 226846
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
    if verbose: print("Filtering selected variables...")
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

