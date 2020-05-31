#!/usr/bin/bash

mkdir -p logs

# Evaluate the best FCN with regression
python -m nicu_los.src.modelling.dnn --evaluation --regression --dropout 0.8 \
  --model-type fcn --hidden-dimension 16 --checkpoint-file \
  fcn_regression_dropout_0_8-batch64-steps1024-epoch04.h5 \

