#!/usr/bin/bash

mkdir -p logs

# FCN -- fine targets
python -m nicu_los.src.modelling.dnn --fine-targets --model-type fcn \
  --model-name fcn_fine_dropout_0_8 --dropout 0.8  --enable-gpu \
  --training-steps 1024 --validation-steps 512 --hidden-dimension 16 \
  --batch-size 64 --allow-growth --epochs 5 
  #--checkpoint-file /home/btstr/NICU-length-of-stay-prediction/models/rnn/checkpoints/fcn_fine_dropout_0_8-batch64-steps1024-epoch05.h5 --initial-epoch 5
