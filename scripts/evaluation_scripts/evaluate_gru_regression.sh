#!/usr/bin/bash

mkdir -p logs

# Evaluate the best GRU regressor
python -m nicu_los.src.modelling.dnn --evaluation --regression --n-cells 1 \
  --dropout 0.3 --model-type gru --checkpoint-file \
  gru_regression_1_cell_dropout_0_3-batch8-steps4096-epoch01.h5 

