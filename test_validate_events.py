from validate_events import validate_events
from datetime import datetime
import unittest
import pandas as pd


class TestEventValidator(unittest.TestCase):
    def setUp(self):
        date_format = "%Y-%m-%d %H:%M:%S"
        self.df_admission = pd.DataFrame(data={'HADM_ID': [1], 'ICUSTAY_ID':
            [3], 'INTIME' : [datetime.strptime("2117-11-20 12:36:10",
                date_format)], 'OUTTIME' : [datetime.strptime(
                    "2117-11-21 14:24:55", date_format)]})

        self.df_events = pd.DataFrame(data={'HADM_ID': [None, None, 1, 1, 2, 1,
            1, 1], 'ICUSTAY_ID': [None, 3, None, 4, 3, 3, 3, 3], 'CHARTTIME' : [
            datetime.strptime("2117-11-20 12:00:00", date_format),
            datetime.strptime("2117-11-20 12:00:00", date_format),
            datetime.strptime("2117-11-20 12:00:00", date_format),
            datetime.strptime("2117-11-20 12:00:00", date_format),
            datetime.strptime("2117-11-20 12:00:00", date_format),
            datetime.strptime("2117-11-20 23:12:01", date_format),
            datetime.strptime("2117-11-20 11:59:59", date_format),
            None]})

        self.stats = {"tot_nb_events" : 0, 'no_charttime': 0,
            'incorrect_charttime': 0, "no_hadm_id_and_icustay_id" : 0,
            "incorrect_hadm_id" : 0, "incorrect_icustay_id" : 0,
            'final_nb_events': 0}

    def test_validate_events(self):
        df_events, stats = validate_events(self.df_admission, self.df_events,
                self.stats)
        self.assertEqual(stats['tot_nb_events'], 8)
        self.assertEqual(stats['no_charttime'], 1)
        self.assertEqual(stats['incorrect_charttime'], 1)
        self.assertEqual(stats['no_hadm_id_and_icustay_id'], 1)
        self.assertEqual(stats['incorrect_hadm_id'], 1)
        self.assertEqual(stats['incorrect_icustay_id'], 1)
        self.assertEqual(stats['final_nb_events'], 3)
        self.assertEqual(len(df_events), 3)
        self.assertEqual(df_events.iloc[0]['HADM_ID'], 1.0)
        self.assertEqual(df_events.iloc[1]['HADM_ID'], 1.0)
        self.assertEqual(df_events.iloc[2]['HADM_ID'], 1.0)
        self.assertEqual(df_events.iloc[0]['ICUSTAY_ID'], 3.0)
        self.assertEqual(df_events.iloc[1]['ICUSTAY_ID'], 3.0)
        self.assertEqual(df_events.iloc[2]['ICUSTAY_ID'], 3.0)


if __name__ == '__main__':
    unittest.main()

