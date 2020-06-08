#!/usr/bin/bash

python -m nicu_los.src.modelling.dnn --evaluation --regression \
  --model-type fcn --checkpoint-file \
  fcn_regression_do_0_5-batch8-steps2048-epoch06.h5  
