from tqdm import tqdm
from sys import argv
import os
import csv
import argparse
import pandas as pd
import numpy as np


def parse_cl_args():
    """Parses CL arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-sp', '--subjects-path', type=str, default='data/',
            help='Path to subject directories.')
    parser.add_argument('-op', '--output-path', type=str,
            default='variable_statistics.csv', help='Path to the output file.')
    parser.add_argument('-v', '--verbose', type=int,
            help='Level of verbosity in console output.', default=1)

    return parser.parse_args(argv[1:])


def main(args):
    verbose, subjects_path = args.verbose, args.subjects_path
    subject_directories = os.listdir(subjects_path)
    subject_directories = set(filter(lambda x: str.isdigit(x),
        subject_directories))
    total_subjects = len(subject_directories)
    subjects_with_vals = 0

    valid_ranges = {
        'Bilirubin -- Direct': {'min': 0.0, 'max': 30.0},
        'Bilirubin -- Indirect': {'min': 0.0, 'max': 30.0},
        'Blood Pressure -- Diastolic': {'min': 0.0, 'max': 100.0},
        'Blood Pressure -- Systolic': {'min': 0.0, 'max': 170.0},
        'Capillary Refill Rate': {'min': 0.0, 'max': 1.0},
        'Fraction Inspired Oxygen': {'min': 21.0, 'max': 100.0}, # impute 21
        'Heart Rate': {'min': 0.0, 'max': 400.0},
        'Height': {'min': 20.0, 'max': 70.0},
        'Oxygen Saturation': {'min': 0.0, 'max': 100.0}, # impute 98
        'pH': {'min': 6.5, 'max': 8.0},
        'Respiratory Rate': {'min': 0.0, 'max': 150.0},
        'Temperature': {'min': 25.0, 'max': 40.0},
        'Weight': {'min': 0.4, 'max': 7.0}
    }

    value_counts = {
        'Bilirubin -- Direct':
            {'values': [], 'Subjects': 0, 'Outliers': 0},
        'Bilirubin -- Indirect':
            {'values': [], 'Subjects': 0, 'Outliers': 0},
        'Blood Pressure -- Diastolic':
            {'values': [], 'Subjects': 0, 'Outliers': 0},
        'Blood Pressure -- Systolic':
            {'values': [], 'Subjects': 0, 'Outliers': 0},
        'Capillary Refill Rate':
            {'values': [], 'Subjects': 0, 'Outliers': 0},
        'Fraction Inspired Oxygen':
            {'values': [], 'Subjects': 0, 'Outliers': 0},
        'Heart Rate':
            {'values': [], 'Subjects': 0, 'Outliers': 0},
        'Height':
            {'values': [], 'Subjects': 0, 'Outliers': 0},
        'Oxygen Saturation':
            {'values': [], 'Subjects': 0, 'Outliers': 0},
        'pH':
            {'values': [], 'Subjects': 0, 'Outliers': 0},
        'Respiratory Rate':
            {'values': [], 'Subjects': 0, 'Outliers': 0},
        'Temperature':
            {'values': [], 'Subjects': 0, 'Outliers': 0},
        'Weight':
            {'values': [], 'Subjects': 0, 'Outliers': 0}
    }

    for i, subject_dir in enumerate(tqdm(subject_directories)):
        # Create empty dataframe
        df = pd.DataFrame(columns=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID',
            'CHARTTIME', 'ITEMID', 'VALUE', 'VALUEUOM', 'VARIABLE'])

        for var in valid_ranges.keys():
            df_events = pd.read_csv(os.path.join(subjects_path,
                str(subject_dir), 'events.csv'))
            df_events = df_events[df_events.VARIABLE == var]

            # Originial number of events
            size_df_orig = len(df_events)

            # Remove invalid values
            df_events = df_events.loc[(df_events.VALUE >=
                valid_ranges[var]['min']) & (df_events.VALUE <=
                    valid_ranges[var]['max'])]
            num_invalid_values = size_df_orig - len(df_events)

            # Store all valid values
            value_counts[var]['values'].extend(df_events.VALUE.tolist())

            # Append to df
            df = df.append(df_events)

            if not df_events.empty:
                value_counts[var]['Subjects'] += 1
                value_counts[var]['Invalid_values'] += num_invalid_values

        # Write df to events.csv
        df.to_csv(os.path.join(subjects_path, str(subject_dir),
            'events.csv'), index=False)

    # Write the results to a file
    csv_header = ['VARIABLE', 'COUNT', 'SUBJECTS', 'INVALID_VALUES', 'MIN',
            'MAX', 'MEAN', 'MEDIAN']

    # Write results to the file
    with open(args.output_path, 'w') as wf:
        wf.write(','.join(csv_header) + '\n')
        csv_writer = csv.DictWriter(wf, fieldnames=csv_header,
            quoting=csv.QUOTE_MINIMAL)

        for key, val in value_counts.items():
            results = {
                'VARIABLE': key,
                'COUNT': len(value_counts[key]['values']),
                'SUBJECTS': value_counts[key]['Subjects'],
                'INVALID_VALUES': value_counts[key]['Invalid_values'],
                'MIN': np.min(value_counts[key]['values']),
                'MAX': np.max(value_counts[key]['values']),
                'MEAN': np.mean(value_counts[key]['values']),
                'MEDIAN': np.median(value_counts[key]['values'])}

            csv_writer.writerows([results])

    if verbose:
        total_invalid_values = 0

        for key, val in value_counts.items():
            total_invalid_values += value_counts[key]['Invalid_values']
            print(f'{key}:')
            for k, v in val.items():
                if k == 'values':
                    print("\tMin:", np.min(v))
                    print("\tMax:", np.max(v))
                    print("\tMean:", np.mean(v))
                    print("\tMedian:", np.median(v))
                    print("\tTotal:", len(v))
                else:
                    print(f'\t{k}: {v}')
            print()
        print(f'The total number of invalid values is: {total_invalid_values}')


if __name__ == '__main__':
    main(parse_cl_args())

