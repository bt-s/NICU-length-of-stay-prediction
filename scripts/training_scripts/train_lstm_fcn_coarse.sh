#!/usr/bin/bash

mkdir -p logs

# LSTM-FCN -- coarse targets
python -m nicu_los.src.modelling.rnn --coarse-targets --model-type lstm_fcn \
  --model-name lstm_fcn_coarse_hid_dim_16_dropout_0_8 --dropout 0.8 \
  --enable-gpu --training-steps 1024  --validation-steps 512 \
  --hidden-dimension 16 --batch-size 32 