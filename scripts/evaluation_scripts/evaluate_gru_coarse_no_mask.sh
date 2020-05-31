#!/usr/bin/bash

mkdir -p logs

# Evaluate using the best GRU with coarse labels, no mask variables
python -m nicu_los.src.modelling.dnn --evaluation --coarse-targets --n-cells 1 \
  --dropout 0.3 --model-type gru --no-mask-indicator --checkpoint-file \
  gru_coarse_1_cell_dropout_0_3_no_mask-batch8-steps4096-epoch31.h5

