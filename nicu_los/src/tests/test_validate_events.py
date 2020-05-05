#!/usr/bin/python3

"""test_validate_events.py
    - Contains tests for testing the validate_events_and_notes_per_subject()
      function.
"""

__author__ = "Bas Straathof"

from nicu_los.src.preprocessing.process_mimic3_tables import \
        validate_events_and_notes_per_subject
import unittest
import pandas as pd


class TestEventValidator(unittest.TestCase):
    def setUp(self):
        self.df_admission = pd.DataFrame(data={'HADM_ID': [1], 'ICUSTAY_ID':
            [3], 'INTIME' : ["2117-11-20 12:36:10"], 'OUTTIME' : [
                    "2117-11-21 14:24:55"]})

        self.df_events = pd.DataFrame(data={'HADM_ID': [None, None, 1, 1, 2, 1,
            1, 1], 'ICUSTAY_ID': [None, 3, None, 4, 3, 3, 3, 3], 'CHARTTIME' : [
                "2117-11-20 12:00:00", "2117-11-20 12:00:00",
                "2117-11-20 12:00:00", "2117-11-20 12:00:00",
                "2117-11-20 12:00:00", "2117-11-20 23:12:01",
                "2117-11-20 12:00:00", None], 'VALUE': [1, 2, 3, 4, 5, 6, None,
                8], 'VALUEUOM': [1, 2, 3, 4, 5, None, 7, 8]
                })

        self.df_notes = pd.DataFrame(data={'SUBJECT_ID': [], 'HADM_ID': [],
            'CHARTTIME': [], 'CATEGORY': [], 'DESCRIPTION': [], 'ISERROR': [],
            'TEXT': []})


        self.stats = {'events_tot_nb_events': 0, 'events_no_value': 0,
                'events_no_charttime': 0, 'events_incorrect_charttime': 0,
                'events_no_hadm_id_and_icustay_id': 0,
                'events_incorrect_hadm_id': 0, 'events_incorrect_icustay_id': 0,
                'events_final_nb_events': 0, 'notes_tot_nb_notes': 0,
                'notes_no_text': 0, 'notes_incorrect_charttime': 0,
                'notes_incorrect_hadm_id': 0, 'notes_final_nb_notes': 0}

    def test_validate_events(self):
        df_events, df_notes, stats = \
                validate_events_and_notes_per_subject(self.df_admission,
                self.df_events, self.df_notes, self.stats)
        self.assertEqual(stats['events_tot_nb_events'], 8)
        self.assertEqual(stats['events_no_charttime'], 1)
        self.assertEqual(stats['events_no_value'], 1)
        self.assertEqual(stats['events_incorrect_charttime'], 0)
        self.assertEqual(stats['events_no_hadm_id_and_icustay_id'], 1)
        self.assertEqual(stats['events_incorrect_hadm_id'], 1)
        self.assertEqual(stats['events_incorrect_icustay_id'], 1)
        self.assertEqual(stats['events_final_nb_events'], 3)
        self.assertEqual(len(df_events), 3)
        self.assertEqual(df_events.iloc[0]['HADM_ID'], 1.0)
        self.assertEqual(df_events.iloc[1]['HADM_ID'], 1.0)
        self.assertEqual(df_events.iloc[2]['HADM_ID'], 1.0)
        self.assertEqual(df_events.iloc[0]['ICUSTAY_ID'], 3.0)
        self.assertEqual(df_events.iloc[1]['ICUSTAY_ID'], 3.0)
        self.assertEqual(df_events.iloc[2]['ICUSTAY_ID'], 3.0)
        self.assertEqual(df_events.iloc[2]['VALUEUOM'], '')


if __name__ == '__main__':
    unittest.main()

