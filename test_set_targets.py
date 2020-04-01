from utils import los_hours_to_target
import unittest
import pandas as pd


class TestTargetSetter(unittest.TestCase):
    def setUp(self):
        self.test_hours = [0, 5, 24, 29, 48, 53, 72, 77, 96, 101, 120, 125, 144,
                149, 168, 173, 192, 197, 216, 317, 336, 341]

        self.expected_targets = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7,
                7, 8, 8, 8, 8, 9, 9]

    def test_set_targets(self):
        test_targets = []
        for hour in self.test_hours:
            test_targets.append(los_hours_to_target(hour))

        for exp_t, test_t in zip(self.expected_targets,
                test_targets):
            self.assertEqual(exp_t, test_t)


if __name__ == '__main__':
    unittest.main()
