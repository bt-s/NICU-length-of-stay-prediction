#!/usr/bin/bash

mkdir -p logs

python -m nicu_los.src.modelling.rnn --fine-targets --model-type lstm_cw \
  --model-name lstm_cw-fine-do0_0-gdo0_2-hid_dim16-multiplier4 --dropout 0.0 \
  --global-dropout 0.2 --training-steps 1024 --validation-steps 1024 \
  --hidden-dimension 16 --multiplier 4 --batch-size 16 --enable-gpu --allow-growth

