#!/usr/bin/python3

"""embed_notes.py

Script to embed the clinical notes into fixed size vectors, using the BioSentVec
embeddings.
"""

__author__ = "Bas Straathof"


import argparse, json, nltk, os, re, sys
import pandas as pd
import multiprocessing as mp

from gensim.models import KeyedVectors
from nltk import sent_tokenize, word_tokenize
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
    parser.add_argument('--biosentvec-model-path', type=str,
            default='../biosentvec_word.bin',
            help='Path to the BioSentVec model file.')

    return parser.parse_args(sys.argv[1:])


def load_biosentvec_model(path, binary=True):
    print('Loading the BioSentVec word embedding model')
    model = KeyedVectors.load_word2vec_format(path, binary=binary)

    print(model.vectors.shape)
    print((model.index2word))
    print(model.index2word[:20])


def main(args):
    subjects_path  = args.subjects_path
    bsv_model_path = args.biosentvec_model_path

    train_dirs = get_subject_dirs(os.path.join(subjects_path, 'train'))
    test_dirs = get_subject_dirs(os.path.join(subjects_path, 'test'))

    subject_dirs = train_dirs + test_dirs
    subject_dirs = ['data/train/659']

    model = load_biosentvec_model(bsv_model_path)

    set_index2word = set(model.index2word)

    # Out-of-vocabulary words
    oov = dict()
    freq = dict()

    for sd in tqdm(subject_dirs):
        with open(os.path.join(sd, 'sentences.json'), 'w') as f:
            json_notes = json.load(f)
            for note, v in json_notes.items():
                print(v)




if __name__ == '__main__':
    main(parse_cl_args())

