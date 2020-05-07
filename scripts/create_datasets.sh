#!/usr/bin/bash

mkdir -p logs

python -m nicu_los.src.preprocessing.process_mimic3_tables \
  > logs/output_process_mimic3_tables
python -m nicu_los.src.preprocessing.preprocess_events_per_subject \
  > logs/output_preprocess_events_per_subject
python -m nicu_los.src.preprocessing.create_timeseries \
  > logs/output_create_timeseries
python -m nicu_los.src.preprocessing.impute_values \
  > logs/output_impute_values
python -m nicu_los.src.preprocessing.split_dataset \
  > logs/output_split_dataset
python -m nicu_los.src.preprocessing.obtain_normalization_statistics \
  > logs/output_obtain_normalization_statistics
python -m nicu_los.src.preprocessing.normalize_values \
  > logs/output_normalize_values
python -m nicu_los.src.preprocessing.create_baseline_datasets \
  > logs/output_create_baseline_datasets
