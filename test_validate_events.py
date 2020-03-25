from validate_events_and_notes import validate_events_and_notes
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


        self.stats_events = {'tot_nb_events': 0, 'no_value': 0,
                'no_charttime': 0, 'incorrect_charttime': 0,
                'no_hadm_id_and_icustay_id': 0, 'incorrect_hadm_id': 0,
                'incorrect_icustay_id': 0, 'final_nb_events': 0}

        self.stats_notes = {'tot_nb_notes': 0, 'no_text': 0,
                'incorrect_charttime': 0, 'incorrect_hadm_id': 0,
                'final_nb_notes': 0}

    def test_validate_events(self):
        df_events, df_notes, stats_events, stats_notes = \
                validate_events_and_notes(self.df_admission, self.df_events,
                self.df_notes, self.stats_events, self.stats_notes)
        self.assertEqual(stats_events['tot_nb_events'], 8)
        self.assertEqual(stats_events['no_charttime'], 1)
        self.assertEqual(stats_events['no_value'], 1)
        self.assertEqual(stats_events['incorrect_charttime'], 0)
        self.assertEqual(stats_events['no_hadm_id_and_icustay_id'], 1)
        self.assertEqual(stats_events['incorrect_hadm_id'], 1)
        self.assertEqual(stats_events['incorrect_icustay_id'], 1)
        self.assertEqual(stats_events['final_nb_events'], 3)
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

