#!/usr/bin/bash

python -m nicu_los.src.modelling.dnn --coarse-targets --model-type lstm_fcn \
  --model-name lstm_fcn_coarse_do_0_8_0_5 --dropout 0.8 --hidden-dimension 16 \
  --batch-size 8 --training-steps 2048 --validation-steps 4096 --epochs 25
