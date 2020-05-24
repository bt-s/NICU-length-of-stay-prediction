#!/usr/bin/bash

mkdir -p logs

# Logistic regression pre-imputed
python -m nicu_los.src.modelling.logistic_regression --model-name \
  log_reg_coarse_gs_pre_imputed --pre-imputed --testing


