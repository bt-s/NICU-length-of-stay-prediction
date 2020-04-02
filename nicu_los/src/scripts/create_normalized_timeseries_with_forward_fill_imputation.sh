#!/usr/bin/bash

mkdir -p logs

python3.7 -m nicu_los.src.scripts.process_mimic3_tables \
  > logs/output_process_mimic3_tables
python3.7 -m nicu_los.src.scripts.preprocess_events_per_subject \
  > logs/output_preprocess_events_per_subject
python3.7 -m nicu_los.src.scripts.create_timeseries \
  > logs/output_create_timeseries
python3.7 -m nicu_los.src.scripts.impute_values \
  > logs/output_impute_values
python3.7 -m nicu_los.src.scripts.split_train_test \
  > logs/output_split_train_test
python3.7 -m nicu_los.src.scripts.obtain_normalization_statistics \
  > logs/output_obtain_normalization_statistics
python3.7 -m nicu_los.src.scripts.normalize_values \
  > logs/output_normalize_values
