#!/usr/bin/bash

mkdir -p logs

# LSTM-FCN -- regression 
python -m nicu_los.src.modelling.rnn --regression --model-type lstm_fcn \
  --model-name lstm_fcn_regression_hid_dim_16_modified_no_schedule --dropout 0.8 \
  --enable-gpu --training-steps 1024  --validation-steps 512 --allow-growth \
  --hidden-dimension 16 --batch-size 64
