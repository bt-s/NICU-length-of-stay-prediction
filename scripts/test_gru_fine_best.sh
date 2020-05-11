#!/usr/bin/bash

mkdir -p logs

# Testing the best GRU with fine labels
python -m nicu_los.src.modelling.rnn --no-coarse-targets --checkpoint-file gru_fine_2_cell_dropout_0_2-batch8-steps2500-epoch04.h5 --n-cells 2 --dropout 0.2 --testing --model-type gru

