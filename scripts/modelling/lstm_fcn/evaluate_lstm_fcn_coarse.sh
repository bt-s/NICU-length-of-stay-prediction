#!/usr/bin/bash

python -m nicu_los.src.modelling.dnn --evaluation --coarse-targets \
  --checkpoint-file lstm_fcn_coarse_do_0_8_0_5-batch8-steps2048-epoch13.h5
