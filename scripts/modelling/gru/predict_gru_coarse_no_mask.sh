#!/usr/bin/bash

python -m nicu_los.src.modelling.dnn --prediction --model-type \
  gru --coarse-targets --dropout 0.3 --batch-size 256 --no-mask-indicator \
  --checkpoint-file gru_coarse_no_mask_1_cell_do_0_3-batch8-steps2048-epoch07.h5

