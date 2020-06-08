#!/usr/bin/bash

python -m nicu_los.src.modelling.dnn --coarse-targets --model-type lstm \
  --no-gestational-age --model-name lstm_coarse_no_ga_1_cell_do_0_3 --n-cells 1 \
  --dropout 0.3 --batch-size 8 --training-steps 2048 --validation-steps 4096 \
  --epochs 11 
