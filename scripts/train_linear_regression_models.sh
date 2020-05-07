#!/usr/bin/bash

mkdir -p logs

# Linear regression with coarse target labels, pre-imputed
python -m nicu_los.src.modelling.linear_regression --coarse-targets 1 \
  --model-name lin_reg_coarse_gs --pre-imputed 1

# Linear regression with coarse target labels, not pre-imputed
python -m nicu_los.src.modelling.linear_regression --coarse-targets 1 \
  --model-name lin_reg_coarse_gs --pre-imputed 0

# Linear regression with fine target labels, pre-imputed
python -m nicu_los.src.modelling.linear_regression --coarse-targets 0 \
  --model-name lin_reg_fine_gs --pre-imputed 1

# Linear regression with fine target labels, not pre-imputed
python -m nicu_los.src.modelling.linear_regression --coarse-targets 0 \
  --model-name lin_reg_fine_gs --pre-imputed 0

