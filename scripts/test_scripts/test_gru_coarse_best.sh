#!/usr/bin/bash

mkdir -p logs

# Testing the best GRU with fine labels
python -m nicu_los.src.modelling.rnn --coarse-targets --checkpoint-file gru_coarse_1_cell_dropout_0_1-batch8-steps2500-epoch04.h5 --n-cells 1 --dropout 0.1 --testing --model-type gru

