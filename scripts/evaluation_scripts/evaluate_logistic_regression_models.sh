#!/usr/bin/bash

mkdir -p logs

# Logistic regression pre-imputed coarse targets
python -m nicu_los.src.modelling.logistic_regression --evaluation \
  --model-name log_reg_coarse_gs_pre_imputed --pre-imputed --coarse-targets

# Logistic regression pre-imputed fine targets
#python -m nicu_los.src.modelling.logistic_regression --evaluation \
  #--model-name log_reg_fine_gs_pre_imputed --pre-imputed --fine-targets

# Logistic regression not pre-imputed coarse targets
python -m nicu_los.src.modelling.logistic_regression --evaluation \
  --model-name log_reg_coarse_gs_not_pre_imputed --not-pre-imputed \
  --coarse-targets

# Logistic regression not pre-imputed fine targets
#python -m nicu_los.src.modelling.logistic_regression -evaluation \
  #--model-name log_reg_fine_gs_not_pre_imputed --not-pre-imputed \
  #--fine-targets

