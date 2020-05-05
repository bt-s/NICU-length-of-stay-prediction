#!/usr/bin/bash

python3.7 -m nicu_los.src.tests.test_ga_extractor
python3.7 -m nicu_los.src.tests.test_round_up_to_hour
python3.7 -m nicu_los.src.tests.test_set_targets
python3.7 -m nicu_los.src.tests.test_transfer_filter
python3.7 -m nicu_los.src.tests.test_validate_events
