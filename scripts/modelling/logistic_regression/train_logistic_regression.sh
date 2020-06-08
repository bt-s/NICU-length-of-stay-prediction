#!/usr/bin/bash

python -m nicu_los.src.modelling.logistic_regression --coarse-targets \
  --no-grid-search --model-name log_reg_coarse_gs_pre_imputed --pre-imputed \
  --C 0.0001 --regularizer l2

python -m nicu_los.src.modelling.logistic_regression --coarse-targets \
  --no-grid-search --model-name log_reg_coarse_gs_not_pre_imputed \
  --not-pre-imputed --C 0.0001 --regularizer l2
