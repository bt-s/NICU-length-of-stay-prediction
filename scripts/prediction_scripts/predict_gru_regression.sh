#!/usr/bin/bash

mkdir -p logs

# Make predictions using the best GRU regressor
python -m nicu_los.src.modelling.dnn --prediction --regression --n-cells 1 \
  --dropout 0.3 --model-type gru --batch-size 1024 --checkpoint-file \
  gru_regression_1_cell_dropout_0_3-batch8-steps4096-epoch01.h5

