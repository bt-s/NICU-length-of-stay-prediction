#!/usr/bin/python3

"""preprocess_notes_per_subject.py

Script to preprocess the clinical notes:
    - Lowercase
    - Replace anonymized piece of note by space
    - Replace underscores by space
    - Tokenize the sentences in the note
    - Tokenize and join the words in the sentences
    - Remove sentences that are shorter than two words
"""

__author__ = "Bas Straathof"


import argparse, json, nltk, os, re, sys
import pandas as pd
import multiprocessing as mp

from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation

from collections import OrderedDict
from itertools import repeat
from tqdm import tqdm

from nicu_los.src.utils.utils import get_subject_dirs, istarmap
from nicu_los.src.utils.reg_exps import reg_exps


def parse_cl_args():
    """Parses CL arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-sp', '--subjects-path', type=str, default='data',
            help='Path to subject directories.')

    return parser.parse_args(sys.argv[1:])


def process_note(note):
    """Process a note into a list of words

    Args:
        note (str): Clinical note

    Returns;
        sentences (list): List of processed sentences extracted from note
    """
    stop_words = set(stopwords.words('english'))

    # Replace anonymized piece of note by space
    note = re.sub(reg_exps['re_anon'], ' ', note)

    # Make some other replacements
    note = note.replace('_', ' ')
    note = note.replace('/', ' / ')
    note = note.replace('.-', ' .- ')
    note = note.replace('.', ' . ')
    note = note.replace('\'', ' \' ')

    # Lowercase the note
    note = note.lower()

    # Tokenize the note
    tokens = [token for token in word_tokenize(note) if token not in \
            punctuation and token not in stop_words]

    # Join the tokens
    note = ' '.join(tokens)

    return note


def clean_note_df(df):
    """Make sure that the dataframe only contains valid entries

    Args:
        df (pd.DataFrame): Pandas DataFrame containing the notes

    Results:
        df (pd.DataFrame): Validated pandas DataFrame containing the notes
    """
    df = df[['HADM_ID', 'CHARTTIME', 'TEXT']]
    df = df[df.HADM_ID.notnull()]
    df = df[df.CHARTTIME.notnull()]
    df = df[df.TEXT.notnull()]

    return df


def process_notes_for_subject(subject_dir):
    """Process the notes corresponding to a subject's NICU

    Saves the processed notes in JSON format were the key is the charttime of
    a unique note, and the value a list of lowercase words extracted from that
    note.

    Args:
        subject_dir (str): Path to the subject directory
    """
    df_notes = pd.read_csv(os.path.join(subject_dir, 'notes.csv'))
    df_notes = clean_note_df(df_notes)

    json_notes = {}
    for index, row in df_notes.iterrows():
        note = process_note(row.TEXT)
        json_notes[f"{row['CHARTTIME']}"] = note

        json_notes = OrderedDict(sorted(json_notes.items(),
            key=lambda t: t[0]))

    with open(os.path.join(subject_dir, 'notes.json'), 'w') as f:
        json.dump(json_notes, f)


def main(args):
    subjects_path  = args.subjects_path

    train_dirs = get_subject_dirs(os.path.join(subjects_path, 'train'))
    test_dirs = get_subject_dirs(os.path.join(subjects_path, 'test'))

    subject_dirs = train_dirs + test_dirs
    subject_dirs = ['data/train/659']

    with mp.Pool() as pool:
        for _ in tqdm(pool.istarmap(process_notes_for_subject,
            zip(subject_dirs)), total=len(subject_dirs)):
            pass


if __name__ == '__main__':
    main(parse_cl_args())

