import pandas as pd
import os
import argparse
import math
from tqdm import tqdm
from sys import argv

import multiprocessing
from itertools import product


def parse_cl_args():
    """Parses CL arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-sp', '--subjects-path', type=str, default='data/',
            help='Path to subject directories.')
    parser.add_argument('-v', '--verbose', type=int,
            help='Level of verbosity in console output.', default=1)

    return parser.parse_args(argv[1:])


ga_weight_map = {
    # Source: https://reader.elsevier.com/reader/sd/pii/S0022347669802246?
    # token=B91F5EB99522C671330FF83AFF557D94481387901AD960AC8E2C7BCFCC3D6124414
    # EF90E94058E0BE5223F53CD9D654A # Table 3
    22: 0.6, # ? g - extrapolated
    23: 0.7, # ? g - extrapolated
    24: 0.8, # ? g - extrapolated
    25: 0.9, # 850 g
    26: 0.9, # 933 g
    27: 1.0, # 1,016 g
    28: 1.1, # 1,113 g
    29: 1.2, # 1,228 g
    30: 1.4, # 1,373 g
    31: 1.5, # 1,540 g
    32: 1.7, # 1,727 g
    33: 1.9, # 1,900 g
    34: 2.1, # 2,113 g
    35: 2.3, # 2,347 g
    36: 2.6, # 2,589 g
    37: 2.9, # 2,868 g
    38: 3.1, # 3,133 g
    39: 3.4, # 3,360 g
    40: 3.5, # 3,480 g
    41: 3.6, # 3,567 g
    42: 3.5, # 3,513 g
    43: 3.4, # 3,416 g
    44: 3.4, # 3,384 g
}

ga_height_map = {
    # Source: https://reader.elsevier.com/reader/sd/pii/S0022347669802246?
    # token=B91F5EB99522C671330FF83AFF557D94481387901AD960AC8E2C7BCFCC3D6124414
    # EF90E94058E0BE5223F53CD9D654A # Table 3
    22: 31.6, # extrapolated
    23: 32.6, # extrapolated
    24: 33.6, # extrapolated
    25: 34.6,
    26: 35.6,
    27: 36.6,
    28: 37.6,
    29: 38.6,
    30: 39.9,
    31: 41.1,
    32: 42.4,
    33: 43.7,
    34: 45.0,
    35: 46.2,
    36: 47.4,
    37: 48.6,
    38: 49.8,
    39: 50.7,
    40: 51.2,
    41: 51.7,
    42: 51.5,
    43: 51.3,
    44: 51.0
}

normal_values = {
    'Bilirubin -- Direct': 0.3,
    'Bilirubin -- Indirect': 6.0,
    'Blood Pressure -- Diastolic': 37.0,
    'Blood Pressure -- Systolic': 67.0,
    'Capillary Refill Rate': 0.0,
    'Fraction Inspired Oxygen': 21.0,
    'Heart Rate': 156.0,
    'Height': ga_height_map,
    'Oxygen Saturation': 97.0,
    'pH': 7.3,
    'Respiratory Rate': 48.0,
    'Temperature': 36.2,
    'Weight': ga_weight_map
}

def impute(sd, variables=list(normal_values.keys()), write_to_file=False):
    ts = pd.read_csv(os.path.join(sd, 'timeseries.csv'))
    # Make sure that the first row contains values such that we can
    # do a forward fill impute
    for var in variables:
        if math.isnan(ts[var].iloc[0]):
            if var == 'Weight' or var == 'Height':
                ts[var].iloc[0] = normal_values[var] \
                        [ts['Gestational Age -- Weeks'].iloc[0]]
            else:
                ts[var].iloc[0] = normal_values[var]

    ts = ts.fillna(method='ffill')

    if write_to_file:
        ts.to_csv(os.path.join(sd, 'timeseries.csv'), index=False)


    return ts


def main(args):
    verbose, subjects_path = args.verbose, args.subjects_path
    subject_directories = os.listdir(subjects_path)
    subject_directories = set(filter(lambda x: str.isdigit(x),
        subject_directories))
    subject_directories = [subjects_path + sd for sd in subject_directories]

    with multiprocessing.Pool() as p:
        max_ = len(subject_directories)
        with tqdm(total=max_) as pbar:
            for i, _ in enumerate(p.imap_unordered(impute, subject_directories)):
                pbar.update()


        #ts.to_csv(os.path.join(subjects_path, str(subject_dir),
            #'timeseries.csv'), index=False)

if __name__ == '__main__':
    main(parse_cl_args())

