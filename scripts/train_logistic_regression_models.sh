#!/usr/bin/bash

mkdir -p logs

# Logistic regression grid search with coarse target labels, pre-imputed
python -m nicu_los.src.modelling.logistic_regression --coarse-targets 1 \
  --grid-search 1 --model-name log_reg_coarse_gs_pre_imputed --pre-imputed 1

# Logistic regression grid search with coarse target labels, not pre-imputed
python -m nicu_los.src.modelling.logistic_regression --coarse-targets 1 \
  --grid-search 1 --model-name log_reg_coarse_gs_not_pre_imputed --pre-imputed 0

# Logistic regression grid search with fine target labels, pre-imputed
python -m nicu_los.src.modelling.logistic_regression --coarse-targets 0 \
  --grid-search 1 --model-name log_reg_fine_gs_pre_imputed --pre-imputed 1

# Logistic regression grid search with fine target labels, not pre-imputed
python -m nicu_los.src.modelling.logistic_regression --coarse-targets 0 \
  --grid-search 1 --model-name log_reg_fine_gs_not_pre_imputed --pre-imputed 0

