#!/usr/bin/bash

python -m nicu_los.src.modelling.dnn --prediction --regression \
  --model-type lstm --dropout 0.3 --batch-size 256 --checkpoint-file \
  lstm_regression_1_cell_do_0_3-batch8-steps2048-epoch26.h5  
  

