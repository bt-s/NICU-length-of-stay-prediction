#!/usr/bin/bash

mkdir -p logs

# Logistic regression pre-imputed coarse targets
python -m nicu_los.src.modelling.logistic_regression --model-name \
  log_reg_coarse_gs_pre_imputed --pre-imputed --testing --coarse-targets

# Logistic regression pre-imputed fine targets
python -m nicu_los.src.modelling.logistic_regression --model-name \
  log_reg_fine_gs_pre_imputed --pre-imputed --testing --fine-targets

# Logistic regression not pre-imputed coarse targets
python -m nicu_los.src.modelling.logistic_regression --model-name \
  log_reg_coarse_gs_not_pre_imputed --not-pre-imputed --testing \
  --coarse-targets

# Logistic regression not pre-imputed fine targets
python -m nicu_los.src.modelling.logistic_regression --model-name \
  log_reg_fine_gs_not_pre_imputed --not-pre-imputed --testing \
  --fine-targets
