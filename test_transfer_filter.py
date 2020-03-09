from utils import transfer_filter
from reg_exps import re_trans_filter
import unittest

test_strings = [
    'discharge diagnosis:   to level iii nicu',
    'discharge disposition: to level 3, [** hospital3 1810 **] neonatal ' +
    'intensive care unit',
    'discharge disposition: level iii nursery at [**hospital **]',
    'discharge disposition: to [**hospital3 1810 **] newborn intensive care '+
    'unit',
    'discharge disposition: transfer to [**hospital3 1810 **] level iii ' +
    'neonatal intensive care unit',
    'discharge disposition: to a level 3 [**hospital 10908 **] due to a lack ' +
    'of available beds at',
    'discharge disposition: to level 3, [**hospital 3  1810 **]',
    'transferred to nicu due to lack of bed availability at [** hospital1' +
    '241 **]',
    'now being transferred to [** hospital1 241 **] due to lack of bed ' +
    'availability',
    'due to lack of bed availability the infant will be transffered to [** ' +
    'hospital3 1810 **]',
    'being transferred to [** hospital1 **] nicu due to lack of beds at ' +
    '[**hospital1 241 **] nicu',
    'discharge disposition: transfer to [** hospital3 1810 **] level iii ' +
    'neonatal intensive care unit',
    'discharge disposition: to [**hospital3 1810 **] newborn intensive care ' +
    'unit',
    'discharge disposition: to level 3 facility',
    'transfer to [**hospital 106 **] for census reasons',
    'patient is to be transferred to the [** hospital 4415 **] nicu due to ' +
    'the lack of available bed space here',
    'discharge disposition:  transferred to [**hospital 10908**], level iii ' +
    'newborn intensive care unit.',
    'patient is to be transferred to [**hospital3 1810**] neonatal intensive ' +
    'care unit care',
    'discharge disposition:  transferred to [**hospital 10908**] level iii ' +
    'newborn intensive care unit.',
    'to be transferred to the [**hospital3 1810**] nicu because of lack of ' +
    'bed space',
    'disposition:  because of the lack of bed space at the [**hospital1 18**]',
    'because of bed unavailability in the nicu, we will transfer the infant',
    'discharge disposition:  to level iii'
]

class TestGAExtractor(unittest.TestCase):
    def test_strings(self):
        for s in test_strings:
            m = transfer_filter(s, re_trans_filter)
            self.assertTrue(m != None)

if __name__ == '__main__':
    unittest.main()

