#!/usr/bin/python3

"""preprocess_notes_per_subject.py

Script to preprocess the clinical events:
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

from nltk import sent_tokenize, word_tokenize
from collections import OrderedDict
from itertools import repeat
from tqdm import tqdm

from nicu_los.src.utils.utils import get_subject_dirs, istarmap
from nicu_los.src.utils.reg_exps import reg_exps


def parse_cl_args():
    """Parses CL arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-op', '--subjects-path', type=str, default='data',
            help='Path to subject directories.')
    parser.add_argument('-v', '--verbose', type=int, default=1,
            help='Info in console output (0 or 1).')

    return parser.parse_args(sys.argv[1:])


def process_note_into_sentences(note):
    """Process a note into a list of sentences

    Args:
        note (str): Clinical note

    Returns;
        sentences (list): List of processed sentences extracted from note
    """
    # Lowercase the note
    note = note.lower()

    # Replace anonymized piece of note by space
    note = re.sub(reg_exps['re_anon'], ' ', note)

    # Replace underscores by space
    note = re.sub(reg_exps['re_under'], ' ', note)

    # Tokenize the sentences in the note
    sentences = sent_tokenize(note)

    # Tokenize and join the words in the sentences
    sentences = [' '.join(word_tokenize(sent)) for sent in sentences]

    # Remove sentences that are shorter than two words
    sentences = [sent for sent in sentences if len(sent) > 2]

    return sentences


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
    """Process all sentences in the notes corresponding to a subject's NICU

    Saves the processed sentences from the notes in JSON format were the
    key is the charttime of a unique note, and the value a list of sentences
    extracted from that note.

    Args:
        subject_dir (str): Path to the subject directory
    """
    df_notes = pd.read_csv(os.path.join(subject_dir, 'notes.csv'))
    df_notes = clean_note_df(df_notes)

    json_sentences = {}
    for index, row in df_notes.iterrows():
        sentences = process_note_into_sentences(row.TEXT)
        json_sentences[f"{row['CHARTTIME']}"] = sentences

        json_sentences = OrderedDict(sorted(json_sentences.items(),
            key=lambda t: t[0]))

    with open(os.path.join(subject_dir, 'sentences.json'), 'w') as f:
        json.dump(json_sentences, f)


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

