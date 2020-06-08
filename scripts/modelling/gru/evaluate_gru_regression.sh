#!/usr/bin/bash

python -m nicu_los.src.modelling.dnn --evaluation --regression \
  --model-type gru --checkpoint-file \
  gru_regression_1_cell_do_0_3-batch8-steps2048-epoch19.h5

