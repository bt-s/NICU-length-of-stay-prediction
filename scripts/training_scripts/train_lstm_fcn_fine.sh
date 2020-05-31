#!/usr/bin/bash

mkdir -p logs

# LSTM-FCN -- fine targets
python -m nicu_los.src.modelling.dnn --fine-targets --model-type lstm_fcn \
  --model-name lstm_fcn_fine_hid_dim_16_modified --dropout 0.8  \
  --enable-gpu --training-steps 1024  --validation-steps 512 --allow-growth \
  --hidden-dimension 16 --batch-size 128 
