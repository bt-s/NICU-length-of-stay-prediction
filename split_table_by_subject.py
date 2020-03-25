import argparse, csv, os, pickle, sys
import pandas as pd
from tqdm import tqdm


def parse_cl_args():
    """Parses CL arguments"""
    parser = argparse.ArgumentParser(
            description='Extract data from the MIMIC-III CSVs.')
    parser.add_argument('-ip', '--input-path', type=str,
            help='Path to MIMIC-III CSV files.')
    parser.add_argument('-op', '--output-path', type=str,
            default='data/', help='Path to output directory.')
    parser.add_argument('-tn', '--table-names', type=str, nargs='+',
            help='Name of the MIMIC-III tables to be read.')
    parser.add_argument('-v', '--verbose', type=int,
            help='Level of verbosity in console output.', default=1)

    return parser.parse_args(sys.argv[1:])


def read_and_split_table_by_subject(mimic_iii_path, table_name, output_path,
    subjects_to_keep=None, verbose=0):
    rows_written = 0

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
        try:
            os.makedirs(subject_f)
        except:
            pass

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

        if verbose: print(f'Read {table_name.upper()}.csv...')
        for i, row in enumerate(tqdm(csv_reader,
            total = rows_per_table[table_name])):
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


def main(args):
    try:
        with open('data/subjects_list.pkl', 'rb') as f:
            subjects = pickle.load(f)
    except IOError:
        print('The file data/subjects.pkl does not exist.')
        raise

    for tn in args.table_names:
        read_and_split_table_by_subject(args.input_path, tn, args.output_path,
                subjects, args.verbose)


if __name__ == '__main__':
    main(parse_cl_args())

