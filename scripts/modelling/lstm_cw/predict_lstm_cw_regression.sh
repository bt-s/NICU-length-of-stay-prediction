#!/usr/bin/bash

python -m nicu_los.src.modelling.dnn --regression --prediction \
  --model-type lstm_cw --dropout 0.0 --global-dropout 0.2 \
 --hidden-dimension 16 --multiplier 4 --batch-size 256 --enable-gpu \
 --allow-growth --checkpoint-file \
  lstm_cw_regression_1_cell_do_0_0_gd_0_2_hd_16_mp_4-batch16-steps1024-epoch09.h5


