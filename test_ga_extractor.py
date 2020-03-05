from utils import extract_gest_age
from reg_exps import reg_exps
import unittest

error_strings = [
    'CO2 19. Gastrointestinal',
    'this string does not contain a gestational age',
    'dol # 4 = 27 2/7 weeks PMA',
    'Now day of life 10, ca [**43**] 5/7 weeks.',
    '~36 [**12-2**] wk CGA BB on DOL 59',
    '~33 [**2-15**] wk CGA BG on DOL 13',
    '38 days old. cga 37.1 weeks',
]


class TestGAExtractor(unittest.TestCase):
    def test_error_strings(self):
        for s in error_strings:
            m, d, w = extract_gest_age(s, reg_exps, verbose=1)
            self.assertEqual([m, d, w], [None, 0, 0])

    def test_strings(self):
        m, d, w = extract_gest_age('day 12 now 41.4', reg_exps)
        self.assertEqual([d, w], [279, 40])

        m, d, w = extract_gest_age('day of life 1 pma 40 [**2-14**]',
                reg_exps)
        self.assertTrue(279 <= d <= 285)
        self.assertTrue(w == 40 or w == 41)

        m, d, w = extract_gest_age('dol #0, cga 35 6/7', reg_exps)
        self.assertEqual([d, w], [251, 36])

        m, d, w = extract_gest_age('day 1 cga 37  ', reg_exps)
        self.assertEqual([d, w], [258, 37])

        m, d, w = extract_gest_age('dol #1, cga 35 2/7', reg_exps)
        self.assertEqual([d, w], [246, 35])

        m, d, w = extract_gest_age('dol #7, cga 40 ', reg_exps)
        self.assertEqual([d, w], [273, 39])

        m, d, w = extract_gest_age('ex 28 wk GA', reg_exps)
        self.assertEqual([d, w], [196, 28])

        m, d, w = extract_gest_age('ex-33-5/7 weeks', reg_exps)
        self.assertEqual([d, w], [236, 34])

        m, d, w = extract_gest_age('former 34 3 week', reg_exps)
        self.assertEqual([d, w], [241, 34])

        m, d, w = extract_gest_age('former 34 3/7 week', reg_exps)
        self.assertEqual([d, w], [241, 34])

        m, d, w = extract_gest_age('former 34 3-7 week', reg_exps)
        self.assertEqual([d, w], [241, 34])

        m, d, w = extract_gest_age('former 32 6/7 weeks gestation',
                reg_exps)
        self.assertEqual([d, w], [230, 33])

        m, d, w = extract_gest_age('former 34+ week', reg_exps)
        self.assertTrue(238 <= d <= 244)
        self.assertTrue(w == 34 or w == 35)

        m, d, w = extract_gest_age('former 34+ wks', reg_exps)
        self.assertTrue(238 <= d <= 244)
        self.assertTrue(w == 34 or w == 35)

        m, d, w = extract_gest_age('former 34  wk', reg_exps)
        self.assertEqual([d, w], [238, 34])

        m, d, w = extract_gest_age('product of a 36-3/7 week gestation',
                reg_exps)
        self.assertEqual([d, w], [255, 36])

        m, d, w = extract_gest_age('product of a 34 3/7 wks', reg_exps)
        self.assertEqual([d, w], [241, 34])

        m, d, w = extract_gest_age('product of a 34 week gestation',
                reg_exps)
        self.assertEqual([d, w], [238, 34])

        m, d, w = extract_gest_age('dol 11 / pma 34 2/3 weeks', reg_exps)
        self.assertEqual([d, w], [229, 33])

        m, d, w = extract_gest_age('dol 13 / pma 34-1/7 wks', reg_exps)
        self.assertEqual([d, w], [226, 32])

        m, d, w = extract_gest_age('dol 13 pma 34 1/7 weeks', reg_exps)
        self.assertEqual([d, w], [226, 32])

        m, d, w = extract_gest_age('dol 11 / now 34.2 weeks', reg_exps)
        self.assertEqual([d, w], [229, 33])

        m, d, w = extract_gest_age('dol 11 / cga 34 weeks', reg_exps)
        self.assertEqual([d, w], [227, 32])

        m, d, w = extract_gest_age('dol 7 ex 34 wker, now 35 wks', reg_exps)
        self.assertEqual([d, w], [238, 34])

        m, d, w = extract_gest_age('dol 35 pma 30 weeks', reg_exps)
        self.assertEqual([d, w], [175, 25])

        m, d, w = extract_gest_age('dol 1 pma 34 3/7 weeks', reg_exps)
        self.assertEqual([d, w], [240, 34])

        m, d, w = extract_gest_age('dol #1, cga 34 1/7 weeks.', reg_exps)
        self.assertEqual([d, w], [238, 34])

        m, d, w = extract_gest_age('dol# 10, cga 35 [**5-5**] wk', reg_exps)
        self.assertTrue(235 <= d <= 241)
        self.assertTrue(w == 34)

        m, d, w = extract_gest_age('dol 18 pma 36 6/7 wks', reg_exps)
        self.assertEqual([d, w], [240, 34])

        m, d, w = extract_gest_age('day 10 pma 31 [**2-19**] wks', reg_exps)
        self.assertTrue(207 <= d <= 213)
        self.assertTrue(w == 30)

        m, d, w = extract_gest_age('day 47, pma 36 1', reg_exps)
        self.assertEqual([d, w], [206, 29])

        m, d, w = extract_gest_age('day 25, now 31.6 wk pma', reg_exps)
        self.assertEqual([d, w], [198, 28])

        m, d, w = extract_gest_age('Now day of life 10, ca 26 5/7 weeks.',
                reg_exps)
        self.assertEqual([d, w], [177, 25])

        m, d, w = extract_gest_age('day 18  pma 34 [**5-6**] wks', reg_exps)
        self.assertTrue(220 <= d <= 226)
        self.assertTrue(w == 31 or w == 32)

        m, d, w = extract_gest_age('estimated gestation age of 30 6/7 wks',
                reg_exps)
        self.assertEqual([d, w], [216, 31])

        m, d, w = extract_gest_age(
                'born at 41 and [**2-5**] wk estimated gestation', reg_exps)
        self.assertTrue(287 <= d <= 293)
        self.assertTrue(w == 41 or w == 42)

        m, d, w = extract_gest_age('delivered at 33-2/7 weeks gestation',
                reg_exps)
        self.assertEqual([d, w], [233, 33])

        m, d, w = extract_gest_age('prematurity at 34 5/7 weeks', reg_exps)
        self.assertEqual([d, w], [243, 35])

        m, d, w = extract_gest_age('34 [**2-1**] wks gestation', reg_exps)
        self.assertTrue(238 <= d <= 244)
        self.assertTrue(w == 34 or w == 35)

        m, d, w = extract_gest_age('34 [**6-20**] wk g.a.', reg_exps)
        self.assertTrue(238 <= d <= 244)
        self.assertTrue(w == 34 or w == 35)

        m, d, w = extract_gest_age('35 [**6-7**]-weeks gestation', reg_exps)
        self.assertTrue(245 <= d <= 251)
        self.assertTrue(w == 35 or w == 36)

        m, d, w = extract_gest_age('24.3 week GA', reg_exps)
        self.assertEqual([d, w], [171, 24])

        m, d, w = extract_gest_age('33-week gestation', reg_exps)
        self.assertEqual([d, w], [231, 33])

        m, d, w = extract_gest_age('24-5/7 weeks gestation', reg_exps)
        self.assertEqual([d, w], [173, 25])

        m, d, w = extract_gest_age('37-2/7-week-gestational', reg_exps)
        self.assertEqual([d, w], [261, 37])

        m, d, w = extract_gest_age('33-1/7 gestation', reg_exps)
        self.assertEqual([d, w], [232, 33])

        m, d, w = extract_gest_age('38 week gestational', reg_exps)
        self.assertEqual([d, w], [266, 38])

        m, d, w = extract_gest_age('38 4/7 weeks gestation', reg_exps)
        self.assertEqual([d, w], [270, 39])

        m, d, w = extract_gest_age('31/1  weeks gestation', reg_exps)
        self.assertEqual([d, w], [218, 31])

        m, d, w = extract_gest_age('34.3 wks gestation', reg_exps)
        self.assertEqual([d, w], [241, 34])

        m, d, w = extract_gest_age('and, 3, 12 weeks, 34-5/7-week    gestation',
                reg_exps)
        self.assertEqual([d, w], [243, 35])

        m, d, w = extract_gest_age('30 and 5/7th wks gestation', reg_exps)
        self.assertEqual([d, w], [215, 31])

        m, d, w = extract_gest_age('30 and 5/7 weeks gestation', reg_exps)
        self.assertEqual([d, w], [215, 31])

        m, d, w = extract_gest_age('24 and 4/7 wks, g.a.', reg_exps)
        self.assertEqual([d, w], [172, 25])

        m, d, w = extract_gest_age('34 and 4/7ths week gestational', reg_exps)
        self.assertEqual([d, w], [242, 35])

        m, d, w = extract_gest_age('born at 35-5/7 weeks’\n gestation',
                reg_exps)
        self.assertEqual([d, w], [250, 36])


if __name__ == '__main__':
    unittest.main()

