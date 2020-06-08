#!/usr/bin/bash

python -m nicu_los.src.modelling.dnn --coarse-targets --model-type fcn \
  --model-name fcn_coarse_do_0_3 --dropout 0.3 --batch-size 8 \
  --training-steps 2048 --validation-steps 4096
