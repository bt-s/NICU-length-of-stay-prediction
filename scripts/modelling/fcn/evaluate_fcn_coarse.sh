#!/usr/bin/bash

python -m nicu_los.src.modelling.dnn --evaluation --coarse-targets \
  --checkpoint-file fcn_coarse_do_0_3-batch8-steps2048-epoch09.h5 
