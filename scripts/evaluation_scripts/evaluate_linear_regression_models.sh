#!/usr/bin/bash

mkdir -p logs

# Evaluate linear regression pre-imputed
#python -m nicu_los.src.modelling.linear_regression --evaluation \
  #--model-name lin_reg_pre_imputed --pre-imputed 

# Evaluate linear regression not pre-imputed
#python -m nicu_los.src.modelling.linear_regression --evaluation \
  #--model-name lin_reg_non_pre_imputed --not-pre-imputed 


python -m nicu_los.src.modelling.linear_regression --test\
  --model-name lin_reg_non_pre_imputed --not-pre-imputed 
