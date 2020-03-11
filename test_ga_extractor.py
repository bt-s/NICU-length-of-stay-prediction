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
    '22 ga spinal needle',
    '22 gauge',
    '64. newborn meds given',
    '20. newborn screen has been sent',
    '72. newborn state screen',
    'BM 24 gavage',
    '24, gavaged over',
    '20, gaining weight',
    '30, gaggy \w initial',
    '2455 male, SGA',
    '22 weeks gestational age',
    '43 weeks gestational age',
    '3495 female',
    '32 and 35 weeks gestation',
    '2915 gam',
    '30 6/7 weeks gestation, Born at less than 32 weeks gestation;',
]


class TestGAExtractor(unittest.TestCase):
    def test_error_strings(self):
        for s in error_strings:
            m, d, w = extract_gest_age(s, reg_exps, verbose=0)
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

        m, d, w = extract_gest_age('34 [**2-1**] wks gestation pregnancy',
                reg_exps)
        self.assertTrue(238 <= d <= 244)
        self.assertTrue(w == 34 or w == 35)

        m, d, w = extract_gest_age('34 [**6-20**] wk g.a.', reg_exps)
        self.assertTrue(238 <= d <= 244)
        self.assertTrue(w == 34 or w == 35)

        m, d, w = extract_gest_age('35 [**6-7**]-weeks gestational', reg_exps)
        self.assertTrue(245 <= d <= 251)
        self.assertTrue(w == 35 or w == 36)

        m, d, w = extract_gest_age('24.3 week GA', reg_exps)
        self.assertEqual([d, w], [171, 24])

        m, d, w = extract_gest_age('33-week gestation born', reg_exps)
        self.assertEqual([d, w], [231, 33])

        m, d, w = extract_gest_age('24-5/7 weeks gestation infant', reg_exps)
        self.assertEqual([d, w], [173, 25])

        m, d, w = extract_gest_age('37-2/7-week-gestational', reg_exps)
        self.assertEqual([d, w], [261, 37])

        m, d, w = extract_gest_age('33-1/7 gestational', reg_exps)
        self.assertEqual([d, w], [232, 33])

        m, d, w = extract_gest_age('38 week gestational', reg_exps)
        self.assertEqual([d, w], [266, 38])

        m, d, w = extract_gest_age('38 4/7 weeks gestation triplet', reg_exps)
        self.assertEqual([d, w], [270, 39])

        m, d, w = extract_gest_age('31/1  weeks gestational', reg_exps)
        self.assertEqual([d, w], [218, 31])

        m, d, w = extract_gest_age('34.3 wks gestation pregnancy', reg_exps)
        self.assertEqual([d, w], [241, 34])

        m, d, w = extract_gest_age('34-5/7-week    gestation newborn',
                reg_exps)
        self.assertEqual([d, w], [243, 35])

        m, d, w = extract_gest_age('30 and 5/7th wks gestation male', reg_exps)
        self.assertEqual([d, w], [215, 31])

        m, d, w = extract_gest_age('30 and 5/7 weeks gestation female',
                reg_exps)
        self.assertEqual([d, w], [215, 31])

        m, d, w = extract_gest_age('24 and 4/7 wks, g.a.', reg_exps)
        self.assertEqual([d, w], [172, 25])

        m, d, w = extract_gest_age('34 and 4/7ths week gestational age',
                reg_exps)
        self.assertEqual([d, w], [242, 35])

        m, d, w = extract_gest_age('born at 35-5/7 weeks’\n gestation',
                reg_exps)
        self.assertEqual([d, w], [250, 36])

        m, d, w = extract_gest_age('37 [**1-11**] week female', reg_exps)
        self.assertTrue(259 <= d <= 265)
        self.assertTrue(w == 37 or w == 38)

        m, d, w = extract_gest_age('twenty-seven and [**3-3**] week twin',
                reg_exps)
        self.assertTrue(189 <= d <= 195)
        self.assertTrue(w == 27 or w == 28)

        m, d, w = extract_gest_age('28-5/7 week premature', reg_exps)
        self.assertEqual([d, w], [201, 29])

        m, d, w = extract_gest_age('GA 34 [**2-16**] wks twin', reg_exps)
        self.assertTrue(238 <= d <= 244)
        self.assertTrue(w == 34 or w == 35)

        m, d, w = extract_gest_age('29 week female', reg_exps)
        self.assertEqual([d, w], [203, 29])

        m, d, w = extract_gest_age('40-2/7-\nweek g.a.', reg_exps)
        self.assertEqual([d, w], [282, 40])

        m, d, w = extract_gest_age('delivered at 28 and 5/7 week', reg_exps)
        self.assertEqual([d, w], [201, 29])

        m, d, w = extract_gest_age('36 [**1-23**] week\nmale', reg_exps)
        self.assertTrue(252 <= d <= 258)
        self.assertTrue(w == 36 or w == 37)

        m, d, w = extract_gest_age('born at 38\nand 4/7 weeks', reg_exps)
        self.assertEqual([d, w], [270, 39])

        m, d, w = extract_gest_age('29 week newborn triplet', reg_exps)
        self.assertEqual([d, w], [203, 29])

        m, d, w = extract_gest_age('former\n25\nand [**2-2**] week', reg_exps)
        self.assertTrue(175 <= d <= 181)
        self.assertTrue(w == 25 or w == 26)

        m, d, w = extract_gest_age('gestation at 28 and 5/7 weeks', reg_exps)
        self.assertEqual([d, w], [201, 29])

        m, d, w = extract_gest_age(
                'born at an estimated gestational age of 34 and 2/7 weeks',
                reg_exps)
        self.assertEqual([d, w], [240, 34])

        m, d, w = extract_gest_age('34 6/7 weeks female', reg_exps)
        self.assertEqual([d, w], [244, 35])

        m, d, w = extract_gest_age('26 and [**5-21**]-week, twin', reg_exps)
        self.assertTrue(182 <= d <= 188)
        self.assertTrue(w == 26 or w == 27)

        m, d, w = extract_gest_age('a former 29 and [**5-14**]\nweek male',
                reg_exps)
        self.assertTrue(203 <= d <= 209)
        self.assertTrue(w == 29 or w == 30)

        m, d, w = extract_gest_age('product of a 33 and [**2-5**] week',
                reg_exps)
        self.assertTrue(231<= d <= 237)
        self.assertTrue(w == 33 or w == 34)

        m, d, w = extract_gest_age('36 6/7 weeks 2225 grams', reg_exps)
        self.assertEqual([d, w], [258, 37])

        m, d, w = extract_gest_age('30-4/7 weeks 1275 gram', reg_exps)
        self.assertEqual([d, w], [214, 31])

        m, d, w = extract_gest_age('32-6/7 weeks 2045 gram', reg_exps)
        self.assertEqual([d, w], [230, 33])

        m, d, w = extract_gest_age('gestational age is 40 and 3/7th weeks',
            reg_exps)
        self.assertEqual([d, w], [283, 40])

        m, d, w = extract_gest_age('Thirty-six and [**12-21**] week GA',
                reg_exps)
        self.assertTrue(252 <= d <= 258)
        self.assertTrue(w == 36 or w == 37)

        m, d, w = extract_gest_age('34 [**5-28**] week infant', reg_exps)
        self.assertTrue(238 <= d <= 244)
        self.assertTrue(w == 34 or w == 35)

        m, d, w = extract_gest_age('40-\n2/7ths week infant', reg_exps)
        self.assertEqual([d, w], [282, 40])

        m, d, w = extract_gest_age('24-5/7th week [**Name', reg_exps)
        self.assertEqual([d, w], [173, 25])

        m, d, w = extract_gest_age('29-6/7\nweeks [**Name', reg_exps)
        self.assertEqual([d, w], [209, 30])

        m, d, w = extract_gest_age('29-6/7\nweeks ga', reg_exps)
        self.assertEqual([d, w], [209, 30])

        m, d, w = extract_gest_age('in labor at 34-6/7 weeks', reg_exps)
        self.assertEqual([d, w], [244, 35])

        m, d, w = extract_gest_age(
                'spotting at 22 weeks GA, born at 39 and 0/7 weeks gestation',
                reg_exps)
        self.assertEqual([d, w], [273, 39])

        m, d, w = extract_gest_age(
                'born at 39 and 0/7 weeks gestation, spotting at 22 weeks GA',
                reg_exps)
        self.assertEqual([d, w], [273, 39])


        m, d, w = extract_gest_age(
                'at 33-5/7 weeks GA, at 33-5/78 weeks gestational age',
                reg_exps)
        self.assertEqual([d, w], [236, 34])

        m, d, w = extract_gest_age('31 and [**3-25**] week baby girl', reg_exps)
        self.assertTrue(217 <= d <= 223)
        self.assertTrue(w == 31 or w == 32)

        m, d, w = extract_gest_age(
                'delivery of infant at 30 6/7 weeks gestation', reg_exps)
        self.assertEqual([d, w], [216, 31])

        m, d, w = extract_gest_age('born at 34 and 3 weeks gestation',
                reg_exps)
        self.assertEqual([d, w], [241, 34])

        m, d, w = extract_gest_age(
                '[200~36 [**6-12**] week gestation male infant',
                reg_exps)
        self.assertTrue(252 <= d <= 258)
        self.assertTrue(w == 36 or w == 37)

        m, d, w = extract_gest_age('24 and [**5-14**] week premature twin',
                reg_exps)
        self.assertTrue(168 <= d <= 174)
        self.assertTrue(w == 24 or w == 25)

        m, d, w = extract_gest_age('DOL 7 / PMA 25-5/7', reg_exps)
        self.assertEqual([d, w], [173, 25])

        m, d, w = extract_gest_age('Prematurity 36 weeks gestation', reg_exps)
        self.assertEqual([d, w], [252, 36])

        m, d, w = extract_gest_age('38 week gestation female twin', reg_exps)
        self.assertEqual([d, w], [266, 38])

        m, d, w = extract_gest_age(
                'Premature female infant at 36 weeks gestational', reg_exps)
        self.assertEqual([d, w], [252, 36])

        m, d, w = extract_gest_age('a 40 week gestation born to', reg_exps)
        self.assertEqual([d, w], [280, 40])

        m, d, w = extract_gest_age('born at approximately 33 weeks gestation',
                reg_exps)
        self.assertEqual([d, w], [231, 33])

        m, d, w = extract_gest_age('39 week gestation) female newborn',
                reg_exps)
        self.assertEqual([d, w], [273, 39])

        m, d, w = extract_gest_age('28 week gestation, female triplet #3',
                reg_exps)
        self.assertEqual([d, w], [196, 28])

        m, d, w = extract_gest_age('32 4/7 weeks female newborn', reg_exps)
        self.assertEqual([d, w], [228, 33])

        m, d, w = extract_gest_age('41 week gestation infant', reg_exps)
        self.assertEqual([d, w], [287, 41])


if __name__ == '__main__':
    unittest.main()

