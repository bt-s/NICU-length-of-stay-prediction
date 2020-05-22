#!/usr/bin/bash

mkdir -p logs

# One cell LSTM network -- fine targets
python -m nicu_los.src.modelling.rnn --fine-targets --model-type lstm \
  --model-name lstm_fine_1_cell_dropout_0_3 --n-cells 1 --dropout 0.3 \
  --batch-size 8 --training-steps 4096 --validation-steps 2048 