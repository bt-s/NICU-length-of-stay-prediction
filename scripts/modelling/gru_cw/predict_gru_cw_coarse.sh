#!/usr/bin/bash

python -m nicu_los.src.modelling.dnn --prediction --coarse-targets \
  --model-type gru_cw --n-cells 1 --dropout 0.3 --batch-size 256 \
  --hidden-dimension 16 --multiplier 4 --enable-gpu --allow-growth \
  --checkpoint-file \
   gru_cw_coarse_1_cell_do_0_0_gd_0_2_hd_16_mp_4-batch16-steps1024-epoch09.h5
