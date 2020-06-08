#!/usr/bin/bash

python -m nicu_los.src.modelling.dnn --prediction --model-type \
  gru --regression --no-gestational-age --dropout 0.3 --batch-size 256 \
  --checkpoint-file \
  gru_regression_no_ga_1_cell_do_0_3-batch8-steps2048-epoch19.h5
