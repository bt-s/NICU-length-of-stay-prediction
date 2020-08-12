#!/usr/bin/bash

python -m nicu_los.src.modelling.dnn --prediction --coarse-targets \
  --model-type fcn --dropout 0.3 --batch-size 256 --checkpoint-file \
  fcn_coarse_do_0_3-batch8-steps2048-epoch09.h5 --friedman
