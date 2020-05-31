#!/usr/bin/bash

mkdir -p logs

# Make predictions with linear regression pre-imputed
python -m nicu_los.src.modelling.linear_regression --prediction \
  --model-name lin_reg_pre_imputed --pre-imputed 

# Make predictions with linear regression not pre-imputed
python -m nicu_los.src.modelling.linear_regression --prediction \
  --model-name lin_reg_non_pre_imputed --not-pre-imputed

