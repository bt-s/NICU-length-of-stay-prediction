#!/usr/bin/bash

mkdir -p logs

# Evaluate the best LSTM-FCN with regression 
python -m nicu_los.src.modelling.dnn --evaluation --regression --dropout 0.8 \
  --model-type lstm_fcn --hidden-dimension 16  --checkpoint-file \
  lstm_fcn_regression_hid_dim_16_modified-batch64-steps1024-epoch06.h5
