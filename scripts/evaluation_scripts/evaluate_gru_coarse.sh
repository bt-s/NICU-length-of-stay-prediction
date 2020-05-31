#!/usr/bin/bash

mkdir -p logs

# Evaluate using the best GRU with coarse labels
python -m nicu_los.src.modelling.dnn --evaluation --coarse-targets \
  --checkpoint-file  gru_coarse_1_cell_dropout_0_3-batch8-steps4096-epoch29.h5 \
  --n-cells 1 --dropout 0.3  --model-type gru

