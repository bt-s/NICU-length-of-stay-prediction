#!/usr/bin/bash

python -m nicu_los.src.modelling.dnn --prediction --coarse-targets \
  --model-type lstm_fcn --dropout 0.8 --hidden-dimension 16 --batch-size 256 \
  --checkpoint-file lstm_fcn_coarse_do_0_8_0_5-batch8-steps2048-epoch13.h5
