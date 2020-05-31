#!/usr/bin/bash

mkdir -p logs

# Make predictions using the best LSTM with coarse labels
python -m nicu_los.src.modelling.dnn --prediction --coarse-targets \
  --checkpoint-file lstm_coarse_1_cell_dropout_0_3-batch8-steps4096-epoch14.h5 \
  --n-cells 1 --dropout 0.3 --model-type lstm --batch-size 1024 
