#!/usr/bin/bash

python -m nicu_los.src.modelling.dnn --evaluation --regression \
  --batch-size 256 --friedman \
  --checkpoint-file lstm_fcn_regression_d0_0_8-batch8-steps2048-epoch05.h5
