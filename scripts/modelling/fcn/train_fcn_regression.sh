#!/usr/bin/bash

python -m nicu_los.src.modelling.dnn --regression --model-type fcn \
  --model-name fcn_regression_do_0_5 --dropout 0.5 --batch-size 8 \
  --training-steps 2048 --validation-steps 4096 --epochs 8 

