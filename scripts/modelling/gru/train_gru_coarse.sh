#!/usr/bin/bash

python -m nicu_los.src.modelling.dnn --coarse-targets --model-type gru \
  --model-name gru_coarse_1_cell_do_0_3 --n-cells 1 --dropout 0.3 \
  --batch-size 8 --training-steps 2048 --validation-steps 4096 

