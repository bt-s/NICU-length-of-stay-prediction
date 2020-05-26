#!/usr/bin/bash

mkdir -p logs

# LSTM-FCN -- coarse targets, no mask variables
python -m nicu_los.src.modelling.rnn --coarse-targets --model-type lstm_fcn \
  --model-name lstm_fcn_coarse_hid_dim_16_modified-no_mask --dropout 0.8 \
  --enable-gpu --training-steps 1024  --validation-steps 512 --allow-growth \
  --hidden-dimension 16 --batch-size 64 --no-mask-indicator

