#!/usr/bin/bash

python -m nicu_los.src.modelling.dnn --evaluation --model-type \
  lstm --regression --friedman --checkpoint-file \
  lstm_regression_no_mask_1_cell_do_0_3-batch8-steps2048-epoch23.h5
