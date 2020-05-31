#!/usr/bin/bash

mkdir -p logs

# LSTM-FCN -- coarse targets
python -m nicu_los.src.modelling.dnn --coarse-targets --model-type lstm_fcn \
  --model-name lstm_fcn_coarse_hid_dim_16_modified --dropout 0.8 \
  --training-steps 4096  --validation-steps 2048 --hidden-dimension 16 --batch-size 8
