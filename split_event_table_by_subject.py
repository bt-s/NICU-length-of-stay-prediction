import argparse, csv, os, pickle, sys
import pandas as pd


def parse_cl_args():
    """Parses CL arguments"""
    parser = argparse.ArgumentParser(
            description='Extract data from the MIMIC-III CSVs.')
    parser.add_argument('-ip', '--input-path', type=str,
            help='Path to MIMIC-III CSV files.')
    parser.add_argument('-tn', '--table-names', type=str, nargs='+',
            default=['test'],
            help='Name of the MIMIC-III events tables to be read.')
    parser.add_argument('-v', '--verbose', type=int,
            help='Level of verbosity in console output.', default=1)

    return parser.parse_args(sys.argv[1:])


def read_and_split_events_table_by_subject(mimic_iii_path, table_name,
        output_path, subjects_to_keep=None, verbose=1):

    # Allow the table name to be passed both with lower- and uppercase letters
    table_name = table_name.upper()

    # Create the output directory
    try:
        os.makedirs(output_path)
    except:
        pass

    if table_name not in ['TEST', 'TEST2', 'CHARTEVENTS', 'LABEVENTS',
            'OUTPUT_EVENTS']:
        raise ValueError("Table name must be one of: 'test', " +
            "'test2', 'chartevents', 'labevents', 'outputevents'")
    else:
        rows_per_table = {'TEST': 9999, 'TEST2': 9999, 'CHARTEVENTS': 330712484,
                'LABEVENTS': 27854056, 'OUTPUTEVENTS': 4349219}
        tot_nb_rows = rows_per_table[table_name]

    # Create a header for the new CSV files to be created
    csv_header = ['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'ITEMID', 'VALUE',
            'VALUEUOM']

    def write_row_to_file():
        # Define a filename for file holding the events of current_subject_id
        subject_f = os.path.join(output_path,
                str(current_subject_id + '_events.csv'))

        # Create the file and give it its header if it doesn't exist yet
        if not os.path.exists(subject_f) or not os.path.isfile(subject_f):
            f = open(subject_f, 'w')
            f.write(','.join(csv_header) + '\n')
            f.close()

        # Write current row to the file
        with open(subject_f, 'a') as wf:
            csv_writer = csv.DictWriter(wf, fieldnames=csv_header,
                quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerows(objects_to_write)

    # Create variables to store the objects to write and the current subject ID
    objects_to_write, current_subject_id = [], ''

    with open(os.path.join(mimic_iii_path, table_name + '.csv')) as table:
        # Create an iterative CSV reader that outputs a row to a dictionary
        csv_reader = csv.DictReader(table)

        for i, row in enumerate(csv_reader):
            if verbose and (i % 1000 == 0):
                print(f'Processed {i}/{tot_nb_rows} rows in '
                      f'{table_name.lower()}.csv')

            if subjects_to_keep and (int(row['SUBJECT_ID']) not in
                    subjects_to_keep):
                continue

            row_output = {'SUBJECT_ID': row['SUBJECT_ID'],
                    'HADM_ID': row['HADM_ID'], 'CHARTTIME': row['CHARTTIME'],
                    'ITEMID': row['ITEMID'], 'VALUE': row['VALUE'],
                    'VALUEUOM': row['VALUEUOM']}

            # For efficiency only write row to file if subjects change
            if current_subject_id != '' and \
                    current_subject_id != row['SUBJECT_ID']:
                write_row_to_file()
                objects_to_write = []

            objects_to_write.append(row_output)
            current_subject_id = row['SUBJECT_ID']

        if i == tot_nb_rows:
            write_row_to_file()
            objects_to_write = []

        if verbose:
            print(f'Processed {i+1}/{tot_nb_rows} rows in '
                    f'{table_name.lower()}.csv')


def main(args):
    try:
        with open('data/subjects_list.pkl', 'rb') as f:
            subjects = pickle.load(f)
    except IOError:
        print('The file data/subjects.pkl does not exist.')
        raise

    table_names = args.table_names
    mimic_iii_path, verbose = args.input_path, args.verbose
    output_path = 'data/events_per_subject/'

    for tn in table_names:
        read_and_split_events_table_by_subject(mimic_iii_path, tn,
                output_path, subjects, verbose)


if __name__ == '__main__':
    main(parse_cl_args())

