#!/usr/bin/bash

mkdir -p logs

# Make predictions using the best LSTM-FCN with regression
python -m nicu_los.src.modelling.dnn --prediction --regression --dropout 0.8 \
  --model-type lstm_fcn --batch-size 64 --hidden-dimension 16 \
  --allow-growth --enable-gpu --checkpoint-file \
  lstm_fcn_regression_hid_dim_16_modified-batch64-steps1024-epoch06.h5

