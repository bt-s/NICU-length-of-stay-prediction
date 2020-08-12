#!/usr/bin/bash

python -m nicu_los.src.modelling.dnn --prediction --regression  \
  --model-type fcn --dropout 0.5 --batch-size 256 --friedman \
  --checkpoint-file fcn_regression_do_0_5-batch8-steps2048-epoch06.h5  
