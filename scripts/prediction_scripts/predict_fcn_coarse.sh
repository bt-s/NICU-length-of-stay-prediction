#!/usr/bin/bash

mkdir -p logs

# Make predictions using the best FCN with coarse labels
#python -m nicu_los.src.modelling.dnn --prediction --coarse-targets \
  #--dropout 0.8 --model-type fcn --batch-size 64 --hidden-dimension 16 \
  #--checkpoint-file fcn_coarse_dropout_0_8_modified-batch64-steps1024-epoch09.h5

python -m nicu_los.src.modelling.dnn --prediction --coarse-targets \
  --dropout 0.8 --model-type fcn --batch-size 8 --hidden-dimension 16 \
  --checkpoint-file fcn_coarse_dropout_0_8_modified-batch8-steps4096-epoch16.h5
