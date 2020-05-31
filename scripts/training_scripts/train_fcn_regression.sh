#!/usr/bin/bash

mkdir -p logs

# FCN -- regression 
python -m nicu_los.src.modelling.dnn --regression --model-type fcn \
  --model-name fcn_regression_dropout_0_8 --dropout 0.8  --enable-gpu \
  --training-steps 1024 --validation-steps 512 --hidden-dimension 16 \
  --batch-size 64 --allow-growth
