#!/usr/bin/bash

python -m nicu_los.src.modelling.dnn --evaluation --model-type \
  gru --regression --no-gestational-age --checkpoint-file \
  gru_regression_no_ga_1_cell_do_0_3-batch8-steps2048-epoch19.h5
