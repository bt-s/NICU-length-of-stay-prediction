#!/usr/bin/bash

python -m nicu_los.src.modelling.dnn --prediction --regression --model-type \
  lstm_fcn --dropout 0.8 --hidden-dimension 16 --batch-size 256 \
  --checkpoint-file lstm_fcn_regression_d0_0_8-batch8-steps2048-epoch05.h5
