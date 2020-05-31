#!/usr/bin/bash

mkdir -p logs

# Evaluate the best LSTM-FCN with coarse labels
python -m nicu_los.src.modelling.dnn --evaluation --coarse-targets \
  --dropout 0.8 --model-type lstm_fcn --hidden-dimension 16 \
  --checkpoint-file lstm_fcn_coarse_hid_dim_16_modified-batch64-steps1024-epoch12.h5

