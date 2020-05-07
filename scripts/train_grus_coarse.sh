#!/usr/bin/bash

mkdir -p logs

# One cell GRU networks -- coarse targets
python -m nicu_los.src.modelling.rnn --coarse-targets 1 --model-type gru \
  --model-name gru_coarse_1_cell_dropout_0_0 --n-cells 1 --dropout 0.0
python -m nicu_los.src.modelling.rnn --coarse-targets 1 --model-type gru \
  --model-name gru_coarse_1_cell_dropout_0_1 --n-cells 1 --dropout 0.1
python -m nicu_los.src.modelling.rnn --coarse-targets 1 --model-type gru \
  --model-name gru_coarse_1_cell_dropout_0_2 --n-cells 1 --dropout 0.2
python -m nicu_los.src.modelling.rnn --coarse-targets 1 --model-type gru \
  --model-name gru_coarse_1_cell_dropout_0_3 --n-cells 1 --dropout 0.3

# Two cell GRU networks -- coarse targets
python -m nicu_los.src.modelling.rnn --coarse-targets 1 --model-type gru \
  --model-name gru_coarse_2_cell_dropout_0_0 --n-cells 2 --dropout 0.0
python -m nicu_los.src.modelling.rnn --coarse-targets 1 --model-type gru \
  --model-name gru_coarse_2_cell_dropout_0_1 --n-cells 2 --dropout 0.1
python -m nicu_los.src.modelling.rnn --coarse-targets 1 --model-type gru \
  --model-name gru_coarse_2_cell_dropout_0_2 --n-cells 2 --dropout 0.2
python -m nicu_los.src.modelling.rnn --coarse-targets 1 --model-type gru \
  --model-name gru_coarse_2_cells_dropout_0_3 --n-cells 2 --dropout 0.3

