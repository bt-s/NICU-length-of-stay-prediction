#!/usr/bin/bash

python -m nicu_los.src.modelling.dnn --evaluation --coarse-targets \
  --model-type gru --checkpoint-file \
  gru_coarse_no_mask_1_cell_do_0_3-batch8-steps2048-epoch07.h5

