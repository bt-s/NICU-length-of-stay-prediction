#!/usr/bin/bash

python -m nicu_los.src.modelling.logistic_regression --prediction \
  --model-name log_reg_coarse_gs_pre_imputed --pre-imputed  --coarse-targets \
  --friedman

python -m nicu_los.src.modelling.logistic_regression --prediction \
  --model-name log_reg_coarse_gs_not_pre_imputed --not-pre-imputed  \
  --coarse-targets --friedman

