#!/usr/bin/bash

mkdir -p logs

# Make predictions using the best channel-wise LSTM with regression 
python -m nicu_los.src.modelling.dnn --prediction --regression --n-cells 1 \
  --dropout 0.0 --global-dropout 0.2 --hidden-dimension 16 --multiplier 4 \
  --model-type lstm_cw --batch-size 1 --allow-growth --enable-gpu --checkpoint-file \
  lstm_cw-regression-do0_0-gdo0_2-hid_dim16-multiplier4-batch16-steps1024-epoch21.h5
