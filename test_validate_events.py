from validate_events import validate_events
import unittest
import pandas as pd


class TestEventValidator(unittest.TestCase):
    def setUp(self):
        self.df_admission = pd.DataFrame(data={'HADM_ID': [1], 'ICUSTAY_ID':
            [3]})

        self.df_events = pd.DataFrame(data={'HADM_ID': [None, None, 1, 1, 2],
            'ICUSTAY_ID': [None, 3, None, 4, 3]})

        self.stats = {"tot_nb_events" : 0, "no_hadm_id_and_icustay_id" : 0,
                "incorrect_hadm_id" : 0, "incorrect_icustay_id" : 0,
                'final_nb_events': 0}

    def test_validate_events(self):
        df_events, stats = validate_events(self.df_admission, self.df_events,
                self.stats)
        self.assertEqual(stats['tot_nb_events'], 5)
        self.assertEqual(stats['no_hadm_id_and_icustay_id'], 1)
        self.assertEqual(stats['incorrect_hadm_id'], 1)
        self.assertEqual(stats['incorrect_icustay_id'], 1)
        self.assertEqual(stats['final_nb_events'], 2)
        self.assertEqual(len(df_events), 2)
        self.assertEqual(df_events.iloc[0]['HADM_ID'], 1.0)
        self.assertEqual(df_events.iloc[1]['HADM_ID'], 1.0)
        self.assertEqual(df_events.iloc[0]['ICUSTAY_ID'], 3.0)
        self.assertEqual(df_events.iloc[1]['ICUSTAY_ID'], 3.0)


if __name__ == '__main__':
    unittest.main()

