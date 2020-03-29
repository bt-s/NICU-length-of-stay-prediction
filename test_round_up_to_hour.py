from utils import round_up_to_hour
import datetime
import unittest

class TestRoundUpToHour(unittest.TestCase):
    def setUp(self):
        self.test_times = [
            round_up_to_hour(datetime.datetime(2012,12,31,23,44,59)),
            round_up_to_hour(datetime.datetime(2012,12,31,23,21,59)),
            round_up_to_hour(datetime.datetime(2012,12,31,22,30,00)),
            round_up_to_hour(datetime.datetime(2012,12,31,22,00,1)),
            round_up_to_hour(datetime.datetime(2012,12,31,22,00,00)),
            round_up_to_hour(datetime.datetime(2012,12,31,21,59,59))
        ]

        self.target_times = [
            datetime.datetime(2013,1,1,00,00,00),
            datetime.datetime(2013,1,1,00,00,00),
            datetime.datetime(2012,12,31,23,00,00),
            datetime.datetime(2012,12,31,23,00,00),
            datetime.datetime(2012,12,31,22,00,00),
            datetime.datetime(2012,12,31,22,00,00),
        ]

    def test_round_up_to_hour(self):
        for test, target in zip(self.test_times, self.target_times):
            self.assertEqual(test, target)


if __name__ == '__main__':
    unittest.main()

