#!/usr/bin/bash

mkdir -p logs

# LSTM-FCN -- coarse targets, no gestational age
python -m nicu_los.src.modelling.dnn --coarse-targets --model-type lstm_fcn \
  --model-name lstm_fcn_coarse_hid_dim_16_modified-no_ga --dropout 0.8 \
  --enable-gpu --training-steps 1024  --validation-steps 512 --allow-growth \
  --hidden-dimension 16 --batch-size 64 --no-gestational-age

