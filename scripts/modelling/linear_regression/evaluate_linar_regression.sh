#!/usr/bin/bash

python -m nicu_los.src.modelling.linear_regression --evaluation \
  --model-name lin_reg_pre_imputed --pre-imputed 

python -m nicu_los.src.modelling.linear_regression --evaluation \
  --model-name lin_reg_non_pre_imputed --not-pre-imputed 

