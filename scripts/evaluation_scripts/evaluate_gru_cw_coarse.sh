#!/usr/bin/bash

mkdir -p logs

# Evaluate the best channel-wise GRU with coarse labels
python -m nicu_los.src.modelling.dnn --evaluation --coarse-targets \
  --dropout 0.0 --global-dropout 0.2 --hidden-dimension 16 --multiplier 4 \
  --model-type gru_cw --checkpoint-file \
  gru_cw-coarse-do0_0-gdo0_2-hid_dim16-multiplier4-batch16-steps1024-epoch18.h5 \
