#!/usr/bin/bash

mkdir -p logs

# Linear regression with coarse target labels, pre-imputed
python -m nicu_los.src.modelling.linear_regression --model-name \
  lin_reg_pre_imputed --pre-imputed 1

# Linear regression with coarse target labels, not pre-imputed
python -m nicu_los.src.modelling.linear_regression --model-name \
  lin_reg_non_pre_imputed --pre-imputed 0

