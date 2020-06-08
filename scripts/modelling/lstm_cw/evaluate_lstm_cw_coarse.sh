#!/usr/bin/bash

python -m nicu_los.src.modelling.dnn --evaluation --coarse-targets \
  --model-type lstm_cw --checkpoint-file \
  lstm_cw_coarse_1_cell_do_0_0_gd_0_2_hd_16_mp_4-batch16-steps1024-epoch10.h5
