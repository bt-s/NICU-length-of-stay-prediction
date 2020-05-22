#!/usr/bin/bash

mkdir -p logs

# FCN -- fine targets
python -m nicu_los.src.modelling.rnn --fine-targets --model-type fcn \
  --model-name fcn_fine_dropout_0_8 --dropout 0.8  --enable-gpu \
  --training-steps 4096 --validation-steps 2048 --hidden-dimension 16 \
  --batch-size 8 
