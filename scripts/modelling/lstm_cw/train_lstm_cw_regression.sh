#!/usr/bin/bash

python -m nicu_los.src.modelling.dnn --regression --model-type lstm_cw \
  --model-name lstm_cw_regression_1_cell_do_0_0_gd_0_2_hd_16_mp_4 \
  --dropout 0.0 --global-dropout 0.2 --training-steps 1024 \
  --validation-steps 2048 --hidden-dimension 16 --multiplier 4 \
  --batch-size 16 --lr-scheduler --epochs 15

