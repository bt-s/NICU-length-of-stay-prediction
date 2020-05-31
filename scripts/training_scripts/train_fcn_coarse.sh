#!/usr/bin/bash

mkdir -p logs

# FCN -- coarse targets
python -m nicu_los.src.modelling.dnn --coarse-targets --model-type fcn \
  --model-name fcn_coarse_dropout_0_8_modified --dropout 0.8 \
  --training-steps 1024 --validation-steps 512 --hidden-dimension 16 \
  --batch-size 64 --checkpoint-file \
  /home/btstr/NICU-length-of-stay-prediction/models/rnn/checkpoints/fcn_coarse_dropout_0_8_modified-batch64-steps1024-epoch10.h5 --initial-epoch 10 --epochs 15 
