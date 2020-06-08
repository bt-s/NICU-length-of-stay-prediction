#!/usr/bin/bash

python -m nicu_los.src.modelling.dnn --evaluation --coarse-targets \
 --model-type lstm --checkpoint-file \
  lstm_coarse_1_cell_do_0_3-batch8-steps2048-epoch10.h5  


