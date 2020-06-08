#!/usr/bin/bash

mkdir -p logs

python -m nicu_los.src.modelling.naive_baselines --model-task classification \
  --coarse-targets --model-name naive_baseline_classification_coarse

python -m nicu_los.src.modelling.naive_baselines --model-task regression \
  --model-name naive_baseline_regression

