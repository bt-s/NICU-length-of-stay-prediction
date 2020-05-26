#!/usr/bin/bash

mkdir -p logs

# Linear regression pre-imputed
python -m nicu_los.src.modelling.linear_regression --model-name \
  lin_reg_pre_imputed --pre-imputed --training

# Linear regression not pre-imputed
python -m nicu_los.src.modelling.linear_regression --model-name \
  lin_reg_non_pre_imputed --not-pre-imputed --training

