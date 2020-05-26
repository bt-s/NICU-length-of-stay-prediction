#!/usr/bin/bash

mkdir -p logs

# One cell GRU network -- fine targets
python -m nicu_los.src.modelling.dnn --fine-targets --model-type gru \
  --model-name gru_fine_1_cell_dropout_0_3_new --n-cells 1 --dropout 0.3 \
  --batch-size 8 --training-steps 4096 --validation-steps 2049 

