#!/usr/bin/bash

mkdir -p logs

# Evaluate the best channel-wise GRU with regression
python -m nicu_los.src.modelling.dnn --evaluation --regression \
  --dropout 0.0 --global-dropout 0.2 --hidden-dimension 16 --multiplier 4 \
  --model-type gru_cw --checkpoint-file \
  gru_cw-regression-do0_0-gdo0_2-hid_dim16-multiplier4-batch16-steps1024-epoch24.h5 \
