#!/usr/bin/bash

mkdir -p logs

# One cell GRU network -- coarse targets
python -m nicu_los.src.modelling.dnn --coarse-targets --model-type gru \
  --model-name gru_coarse_1_cell_dropout_0_3 --n-cells 1 --dropout 0.3 \
  --batch-size 8 --training-steps 4096 --validation-steps 2048 \
  --checkpoint-file /home/btstr/NICU-length-of-stay-prediction/models/rnn/checkpoints/gru_coarse_1_cell_dropout_0_3-batch8-steps4096-epoch30.h5 --initial-epoch 30 --epochs 35
