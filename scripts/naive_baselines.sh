#!/usr/bin/bash

mkdir -p logs

# Naive baseline: classification with coarse target labels
python -m nicu_los.src.modelling.naive_baselines --model-task classification \
	--coarse-targets --model-name naive_baseline_classification_coarse

# Naive baseline: classification with fine target labels
python -m nicu_los.src.modelling.naive_baselines --model-task classification \
	--no-coarse-targets --model-name naive_baseline_classification_fine

# Naive baseline: regression 
python -m nicu_los.src.modelling.naive_baselines --model-task regression \
	--model-name naive_baseline_regression

