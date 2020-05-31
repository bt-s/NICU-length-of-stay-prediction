#!/usr/bin/bash

mkdir -p logs

# Make predictions using the best FCN with regression
python -m nicu_los.src.modelling.dnn --prediction --regression \
  --dropout 0.8 --model-type fcn --batch-size 64 --hidden-dimension 16 \
  --allow-growth --enable-gpu --checkpoint-file \
  fcn_regression_dropout_0_8-batch64-steps1024-epoch04.h5

