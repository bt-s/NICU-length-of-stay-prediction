#!/usr/bin/bash

python -m nicu_los.src.modelling.dnn --regression --model-type lstm_fcn \
  --model-name lstm_fcn_regression_d0_0_8 --dropout 0.8 --hidden-dimension 16 \
  --batch-size 8 --training-steps 2048 --validation-steps 4096 
