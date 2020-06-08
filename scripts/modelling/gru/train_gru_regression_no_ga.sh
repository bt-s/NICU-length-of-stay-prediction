#!/usr/bin/bash

python -m nicu_los.src.modelling.dnn --regression --model-type gru \
  --no-gestational-age --model-name gru_regression_no_ga_1_cell_do_0_3 \
 --n-cells 1 --dropout 0.3 --batch-size 8 --training-steps 2048 \
 --validation-steps 4096 --epochs 20

