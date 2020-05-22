#!/usr/bin/bash

mkdir -p logs

# Logistic regression grid search with coarse target labels, pre-imputed
python -m nicu_los.src.modelling.logistic_regression --coarse-targets \
  --grid-search --model-name log_reg_coarse_gs_pre_imputed --pre-imputed

# Logistic regression grid search with coarse target labels, not pre-imputed
python -m nicu_los.src.modelling.logistic_regression --coarse-targets \
  --grid-search --model-name log_reg_coarse_gs_not_pre_imputed \
  --not-pre-imputed

# Logistic regression grid search with fine target labels, pre-imputed
python -m nicu_los.src.modelling.logistic_regression --fine-targets \
  --grid-search --model-name log_reg_fine_gs_pre_imputed --pre-imputed

# Logistic regression grid search with fine target labels, not pre-imputed
python -m nicu_los.src.modelling.logistic_regression --fine-targets \
  --grid-search --model-name log_reg_fine_gs_not_pre_imputed --not-pre-imputed

