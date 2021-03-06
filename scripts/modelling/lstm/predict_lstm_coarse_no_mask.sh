#!/usr/bin/bash

python -m nicu_los.src.modelling.dnn --prediction --model-type \
  lstm --coarse-targets --dropout 0.3 --batch-size 256 --no-mask-indicator \
  --checkpoint-file lstm_coarse_no_mask_1_cell_do_0_3-batch8-steps2048-epoch10.h5

