#!/usr/bin/bash

python -m nicu_los.src.modelling.logistic_regression --evaluation \
  --model-name log_reg_coarse_gs_pre_imputed --pre-imputed --coarse-targets

python -m nicu_los.src.modelling.logistic_regression --evaluation \
  --model-name log_reg_coarse_gs_not_pre_imputed --not-pre-imputed \
  --coarse-targets

