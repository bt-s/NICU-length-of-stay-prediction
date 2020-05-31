#!/usr/bin/bash

mkdir -p logs

# Make predictions using the best LSTM-FCN with coarse labels
python -m nicu_los.src.modelling.dnn --prediction --coarse-targets --checkpoint-file \
  lstm_fcn_coarse_hid_dim_16_modified-batch64-steps1024-epoch12.h5 \
  --dropout 0.8 --model-type lstm_fcn --batch-size 64 \
  --hidden-dimension 16 --enable-gpu --allow-growth


