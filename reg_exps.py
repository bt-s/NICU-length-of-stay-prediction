#!/usr/bin/python3

"""reg_exps.py - Contains regular expressions used in this study.

As part of my Master's thesis at KTH Royal Institute of Technology.
"""

__author__ = "Bas Straathof"


import re


re_ga = re.compile(
        '(((\d{2}|(twenty|thirty|forty)(-| |)?(one|two|three|four|five|six|' +
        'seven|eight|nine)?)( +and +| +- +|-| +)? *(\[\*\*\d{1,2}-\d{1,2}' +
        '\*\*\]|\d{1}|/|\.)?[\/|-]?(\d{1})? *(th|ths)?(-? *| +)((wk|week)s? *' +
        '(infant|\[\*\*name)|(wk|week)?s?(,|-|\’)? *(est.?|estimated)? *(' +
        'triplet|twin|\d{4} gram|(newborn|baby) (boy|girl)|(fe)?male|newborn' +
        '(?! meds given)(?! (state )?screen)|premature|gestation|ga(?!uge)' +
        '(?!ggy)(?!in)(?!vage)(?! spinal needle)(?!stro)|g\.a\.|ga\.)))|' +
        '((former|ex|product of a|delivered at|born at|in labor at|gestation ' +
        'at|gestation(al)? age (of|is)| ga|prematurity at)( +|-)\d{2}\+?(' +
        ' +and +|-)? *(\[\*\*\d{1,2}-\d{1,2}\*\*\]|\d{1}/\d{1}|\d{1}-\d{1}|' +
        '\d{1})?(th|ths)?-? *(week|wk)s? *-?(gestation|ga |g\.a\.|ga\.)?))')

# Regular expression to capture the corrected gestational age of a patient
re_cga = re.compile(
        'd(ol|ay|ay of life|ays old) *#? *(\d{1,2} *(,|/)? *(pma|' +
        'cga|ca|now)) *((\d{2}|(twenty|thirty|forty)(-| |)?(one|two|three|' +
        'four|five|six|seven|eight|nine)?) *-?(\d/\d|\[\*\*\d{1,2}-\d{1,2}' +
        '\*\*\]|.\d| \d)|\d{2})')

# Regex to capture double digits
re_dd = re.compile('\d{2}')

# Regex to caputure one or more digits
re_dol = re.compile('\d+')

# Regex to capture anonymized gestational age strings and digit_digit+
re_anon_dd_p = re.compile('\[\*\*\d|\d{2}\+')

# Regex to capture digit/digit
re_d_d_slash = re.compile('\d{1}/\d{1}')

# Regex to capture digit-digit
re_d_d_dash = re.compile('\d{1}-\d{1}')

# Regex to caputre digitdigit/digit, digitdigit.digit and digitdigit digit
re_dd_d = re.compile('(\d{2}\/\d{1})|(\d{2}\.\d{1})|(\d{2}\ \d{1})')

# Regex to capture common false gestational age
re_false = re.compile('(born at less than 32 weeks gestation|\d{4} (fe)?male|32 and 35 weeks gestation)')

# Regex to split a corrected gestational age regex match string
re_splitter = re.compile('(pma|ca|cga|now)')

re_trans_filter = re.compile(
        '(discharge (diagnosis|disposition): +(transfer(red)? )?(to)?' +
        '((( a)? *level *(iii|3|three),? *(\[\*\* *hospital *\d* *\d+ ' +
        '*\*\*\]|(neonatal|newborn) intensive care unit|nicu|nursery|facility' +
        '|due to a lack of available beds))| +\[\*\* *hospital *\d* *\d+ ' +
        '*\*\*\],? +(level (iii|3|three))? *(nicu|(newborn|neonatal) ' +
        'intensive care unit)| level (iii|3|three))|(due to|because of' +
        '|for) *(bed unavailability|(the )?lack of (bed( availability|s at' +
        '| space)|available bed))|transfer(red)? to \[\*\* *hospital *\d* ' +
        '*\d+ *\*\*\] +((due to|because of|for) * census reasons|nicu|' +
        '(neonatal|newborn) intensive care unit))')

# Dictionary containing all regular expressions
reg_exps = {'re_ga': re_ga, 're_cga': re_cga, 're_dd': re_dd, 're_dol': re_dol,
        're_anon_dd_p': re_anon_dd_p, 're_d_d_slash': re_d_d_slash,
        're_d_d_dash': re_d_d_dash, 're_dd_d': re_dd_d, 're_false': re_false,
        're_splitter': re_splitter, 're_trans_filter': re_trans_filter}

