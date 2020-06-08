#!/usr/bin/bash

python -m nicu_los.src.modelling.dnn --regression  --model-type gru_cw \
  --model-name gru_cw_regression_1_cell_do_0_0_gd_0_2_hd_16_mp_4 --n-cells 1 \
  --dropout 0.0 --global-dropout 0.2 --batch-size 16 --hidden-dimension 16 \
  --multiplier 4 --training-steps 1024 --validation-steps 2048 --epochs 15 \
  --lr-scheduler
