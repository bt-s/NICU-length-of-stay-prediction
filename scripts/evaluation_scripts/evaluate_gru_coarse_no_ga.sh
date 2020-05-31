#!/usr/bin/bash

mkdir -p logs

# Evaluating the best GRU with coarse labels, no gestational age variable
python -m nicu_los.src.modelling.dnn --evaluation --coarse-targets \
  --n-cells 1 --dropout 0.3 --model-type gru --no-gestational-age \
  --checkpoint-file gru_coarse_1_cell_dropout_0_3_no_ga-batch8-steps4096-epoch35.h5

