#!/usr/bin/python3

"""reg_exps.py - Contains regular expressions used in this study.

As part of my Master's thesis at KTH Royal Institute of Technology.
"""

__author__ = "Bas Straathof"


import re


# Regular expression to capture the gestational age of a patient
re_ga = re.compile('(\d{2}( +and +| +- +|-| +)?(\[\*\*\d{1,2}-\d{1,2}\*\*\]' +
        '|\d|/|\.)[\/|-]?\d?(th|ths)?(-?| +)(wk|week)?s?(,|-|\’)? *(est.?|' +
        'estimated)? *(gestation|ga |g\.a\.|ga\.))|(\d{2}((\.|\/|-)\d{1})?' +
        '[ +|-](week|wk)(s)? +(gestation|ga |g.a.|ga))|((former|ex|' +
        'product of a|delivered at|born at)( +|-)\d{2}\+?-? *(\d/\d|\d-\d|\d)' +
        '?-? *(week|wk)s? *-?(gestation|ga |g\.a\.|ga\.)?|((prematurity at|' +
        '(estimated|est|est.) *(gestation(al)? age|ga|g\.a\.|ga) *of) *\d{2} +' +
        '(\d/\d)? *(week|wk)))')

# Regular expression to capture the corrected gestational age of a patient
re_cga = re.compile('d(ol|ay|ay of life|ays old) *#? *(\d{1,2} *(,|/)? ' +
        '*(pma|cga|ca|now)) *(\d{2} *-?(\d/\d|\[\*\*\d{1,2}-\d{1,2}\*\*\]|' +
        '.\d| \d)|\d{2})')

# Regex to capture double digits
re_dd = re.compile('\d{2}')

# Regex to caputure one or more digits
re_dol = re.compile('\d+')

# Regex to capture anonymized gestational age strings and digit_digit+
re_anon_dd_p = re.compile('\[\*\*|\d{2}\+')

# Regex to capture digit/digit
re_d_d_slash = re.compile('\d{1}/\d{1}')

# Regex to capture digit-digit
re_d_d_dash = re.compile('\d{1}-\d{1}')

# Regex to caputre digitdigit/digit, digitdigit.digit and digitdigit digit
re_dd_d = re.compile('(\d{2}\/\d{1})|(\d{2}\.\d{1})|(\d{2}\ \d{1})')

# Regex to split a corrected gestational age regex match string
re_splitter = re.compile('(pma|ca|cga|now)')

# Dictionary containing all regular expressions
reg_exps = {'re_ga': re_ga, 're_cga': re_cga, 're_dd': re_dd, 're_dol': re_dol,
        're_anon_dd_p': re_anon_dd_p, 're_d_d_slash': re_d_d_slash, 're_d_d_dash': re_d_d_dash, 're_dd_d': re_dd_d,
        're_splitter': re_splitter}

