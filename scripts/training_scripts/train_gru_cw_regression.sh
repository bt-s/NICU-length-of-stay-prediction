#!/usr/bin/bash

mkdir -p logs

python -m nicu_los.src.modelling.dnn --regression --model-type gru_cw \
  --model-name gru_cw-regression-do0_0-gdo0_2-hid_dim16-multiplier4 --dropout 0.0 \
  --global-dropout 0.2 --training-steps 1024 --validation-steps 1024 \
  --hidden-dimension 16 --multiplier 4 --batch-size 16 --lr-scheduler \
  --checkpoint-file /home/btstr/NICU-length-of-stay-prediction/models/rnn/checkpoints/gru_cw-regression-do0_0-gdo0_2-hid_dim16-multiplier4-batch16-steps1024-epoch20.h5 --initial-epoch 20 --epochs 25
