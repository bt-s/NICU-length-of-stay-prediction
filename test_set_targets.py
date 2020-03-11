from utils import set_targets
import unittest
import pandas as pd


class TestTargetSetter(unittest.TestCase):
    def setUp(self):
        self.test_df = pd.DataFrame(data={'LOS': [0, 0.2, 1, 1.2, 2, 2.2, 3,
            3.2, 4, 4.2, 5, 5.2, 6, 6.2, 7, 7.2, 8, 8.2, 9, 13.2, 14, 14.2]})

        self.test_df_w_targets = set_targets(self.test_df)

        self.expected_targets = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7,
                7, 8, 8, 8, 8, 9, 9]

    def test_set_targets(self):
        for exp_t, (ix, row) in zip(self.expected_targets,
                self.test_df_w_targets.iterrows()):
            t = int(self.test_df_w_targets.iloc[ix]['TARGET'])
            self.assertEqual(t, exp_t)


if __name__ == '__main__':
    unittest.main()
