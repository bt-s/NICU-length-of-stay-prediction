#!/usr/bin/python3

"""utils.py - Utils for extracting data of newborn patients from the
              MIMIC-III CSVs.

As part of my Master's thesis at KTH Royal Institute of Technology.
"""

__author__ = "Bas Straathof"

import datetime, os, re, shutil, time

import pandas as pd
import numpy as np
import multiprocessing.pool as mpp

from word2number import w2n


def filter_on_newborns(df):
    """Filter df on patients zero days of age

    Args:
        df (pd.DataFrame): Admissions dataframe

    Returns:
        df (pd.DataFrame): Filtered dataframe
    """
    df['AGE'] = (df['ADMITTIME'] - df['DOB']).dt.days
    df = df[df['AGE'] == 0]
    df = df.drop(['AGE'], axis=1)

    return df


def extract_from_cga_match(match_str_cga, reg_exps):
    """Extract the gestational age (GA) in weeks and days from a
    match of the corrected GA (CGA)

    Args:
        match_str_cga (str): String containing the CGA match
        reg_exps (dict): All regular expressions

    Returns:
        days_ga (int): GA in days
        weeks_ga_round (int): Rounded GA in weeks
    """
    match_str = reg_exps['re_splitter'].split(match_str_cga)
    match_str_dol, match_str_cga = match_str[0], match_str[2]

    # Extract the days of life
    dol = int(reg_exps['re_dol'].search(match_str_dol).group(0))

    # Extract the days part of the correct gestational age
    if reg_exps['re_dd_d'].findall(match_str_cga):
        days_cga = int(reg_exps['re_dd_d'].search(match_str_cga)
                .group(0)[-1])
    elif reg_exps['re_anon_dd_p'].findall(match_str_cga):
        # Randomly sample # of days if unknown or anonymized
        days_cga = np.random.choice(np.arange(0, 7))
    elif reg_exps['re_d_d_slash'].findall(match_str_cga):
        days_cga = int(reg_exps['re_d_d_slash'].search(match_str_cga)
                .group(0)[0])
    elif reg_exps['re_d_d_dash'].findall(match_str_cga):
        days_cga = int(reg_exps['re_d_d_dash'].search(match_str_cga)
                .group(0)[0])
    else:
        days_cga = 0

    # Extract weeks from match
    weeks_cga = int(reg_exps['re_dd'].findall(match_str_cga)[0])

    days_ga = weeks_cga*7 + days_cga - dol
    weeks_ga_round = int(round(days_ga/7, 0))

    return days_ga, weeks_ga_round


def extract_from_ga_match(match_str_ga, reg_exps):
    """Extract the gestational age (GA) in weeks and days from a
    direct match of the GA

    Args:
        match_str_ga (str): String containing the GA match
        reg_exps (dict): All regular expressions

    Returns:
        days_ga (int): GA in days
        weeks_ga_round (int): Rounded GA in weeks
    """
    # Extract the days part of the gestational age
    if reg_exps['re_dd_d'].findall(match_str_ga):
        days_ga = int(reg_exps['re_dd_d'].search(match_str_ga)
                .group(0)[-1])
    # Randomly sample # of days if unknown or anonymized
    elif reg_exps['re_anon_dd_p'].findall(match_str_ga):
        days_ga = np.random.choice(np.arange(0, 7))
    elif reg_exps['re_d_d_slash'].findall(match_str_ga):
        days_ga = int(reg_exps['re_d_d_slash'].search(match_str_ga).group(0)[0])
    elif reg_exps['re_d_d_dash'].findall(match_str_ga):
        days_ga = int(reg_exps['re_d_d_dash'].search(match_str_ga).group(0)[0])
    else:
        days_ga = 0

    # Extract weeks from match
    try:
        weeks_ga = w2n.word_to_num(match_str_ga)
    except ValueError:
        weeks_ga = int(reg_exps['re_dd'].findall(match_str_ga)[0])

    # Calculate days GA
    days_ga += weeks_ga*7

    # Round weeks + days
    weeks_ga_round = int(round(days_ga/7, 0))

    return days_ga, weeks_ga_round


def extract_gest_age_from_note(s, reg_exps, verbose=False):
    """Extract the gestational age (GA) in weeks and days from a
    clinical note

    Args:
        s (str): String containing the test of the clinical note
        reg_exps (dict): All regular expressions
        verbose (bool): Verbosity flag

    Returns:
        match_str (str): String of the match containing the (C)GA
        max_days_ga (int): Maximum extracted GA in days
        max_weeks_ga_round (int): Maximum extracted ounded GA in weeks
    """
    # We want to find the maximum reported value in the clinical note
    match_str, max_days_ga, max_weeks_ga_round = None, 0, 0

    # Reformat string to lowercase without new line characters
    s = s.replace('\n', ' ').lower()

    # Filter out false string that occurs in many notes
    s = re.sub(reg_exps['re_false'], '', s)

    # See if a match can be found with the corrected gestational age regex
    # Assumption: if mentioned, the CGA is only mentioned once
    match = reg_exps['re_cga'].search(s)

    if match:
        # Extract string from match
        match_str = match.group(0)

        if not re.match(reg_exps['re_not_allowed'], match_str):
            days_ga, weeks_ga_round = extract_from_cga_match(match_str,
                    reg_exps)
            if (23 < weeks_ga_round < 43):
                max_weeks_ga_round = weeks_ga_round
                max_days_ga = days_ga
            else:
                match_str = None
        else:
            match_str = None
    else:
        # See if matches can be found with the gestational age regex
        matches = reg_exps['re_ga'].findall(s)

        if len(matches) != 0:
            # Extract the match with the highest gestational age
            for m in range(len(matches)):
                if not re.match(reg_exps['re_not_allowed'], matches[m][0]):
                    days_ga, weeks_ga_round = extract_from_ga_match(
                            matches[m][0], reg_exps)
                    if ((weeks_ga_round > max_weeks_ga_round) and
                            (23 < weeks_ga_round < 43)):
                        max_weeks_ga_round = weeks_ga_round
                        max_days_ga = days_ga
                        match_str = matches[m][0]
        else:
            if verbose: print(f'The GA cannot be extracted from: {s}')

    return match_str, max_days_ga, max_weeks_ga_round


def transfer_filter(s, reg_exps, verbose=False):
    """Check if note describes stay with a capacity-related transfer

    Args:
        s (str): Text of the note
        reg_exps (dict): All regular expressions
        verbose (bool): Verbosity flag

    Returns:
        match (re.Match/None): Matched regular expression
    """
    # Default: no match is found
    match = None

    # Reformat string to lowercase without new line characters
    s = s.replace('\n', ' ').lower()

    # See if a match can be found with the unpredictable transfer filter regex
    match = reg_exps['re_trans_filter'].search(s)

    return match


def los_hours_to_target(hours):
    """Convert LOS in hours to targets

    The targets exist of ten buckets:
        0: less than 1 day; 1: 1 day; 2: 2 day; 3: 3 day; 4: 4 day;
        5: 5 day; 6: 6 day; 7: 7 day; 8: 8-13 days; 9: more than 14 days

    Args:
        hours (int): LOS in hours

    Return:
        target (int): The respective target
    """
    if hours < 24:
        target = 0
    elif 24 <= hours < 48:
        target = 1
    elif 48 <= hours < 72:
        target = 2
    elif 72 <= hours < 96:
        target = 3
    elif 96 <= hours < 120:
        target = 4
    elif 120 <= hours < 144:
        target = 5
    elif 144 <= hours < 168:
        target = 6
    elif 168 <= hours < 192:
        target = 7
    elif 192 <= hours < 336:
        target = 8
    elif 336 <= hours:
        target = 9

    return target


def round_up_to_hour(dt):
    """Round datetime up to hour

    Args:
        dt (datetime.datetime/str): Original atetime

    Returns:
        dt (datetime.datetime/str): Datetime rounded up to hour
    """
    if type(dt) == str:
        date_format = "%Y-%m-%d %H:%M:%S"
        dt = datetime.datetime.strptime(dt, date_format)

    hour = 3600

    # Seconds passed in current day
    seconds = (dt - dt.min).seconds

    # Floor division to closest next hour if not whole hour on clock
    rounding = (seconds + hour-1) // hour * hour

    # Use timedelta to set the rounded time
    dt = dt + datetime.timedelta(0, rounding-seconds, -dt.microsecond)

    return dt


def compute_ga_days_for_charttime(charttime, intime, ga_days_birth):
    """Compute the gestational age in days at a specific charttime

    Args:
        charttime (datetime.datetime): The charttime
        intime (datetime.datetime): The admission time of the ICU stay
        ga_days_birth: The gestational age in days at birth

    Returns:
        (int) Gestational age in days
    """
    return round(((charttime - intime).days + ga_days_birth))


def compute_remaining_los(charttime, intime, los_hours_total):
    """Compute the remaining LOS in hours at a specific charttime

    Args:
        charttime (datetime.datetime): The charttime
        intime (datetime.datetime): The admission time of the ICU stay
        los_hours_total: Total LOS in hours

    Returns:
        (int) Remaining LOS in hours
    """
    return round(los_hours_total - (charttime - intime) \
            .total_seconds() // 3600)


def get_first_valid_value_from_ts(ts, variable):
    """Get the first valid value of a variable in a time series

    Args:
        ts (pd.DataFrame): Timeseries dataframe
        variable (str): Name of the variable

    Returns:
        value (float): First valid value of variable
    """
    # Assume no value exists
    value = np.nan
    if variable in ts:
        # Find the indices of the rows where variable has a value in ts
        indices = ts[variable].notnull()
        if indices.any():
            index = indices.to_list().index(True)
            value = ts[variable].iloc[index]

    return value


def timing(f):
    """Custom decorator to time how long it takes to execute a function

    Args:
        f (func): Function to be timed

    Returns:
        wrap (any): Closure
    """
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print(f'{f.__name__} function took {round(time2-time1, 3)} s')

        return ret

    return wrap


def remove_subject_dir(path):
    """Remove the directory of a subject

    Args:
        path (str): Path to the directory of the subject
    """
    try:
        shutil.rmtree(path)
    except OSError as e:
        print (f'Error: {e.filename} - {e.strerror}.')


def istarmap(self, func, iterable, chunksize=1):
    """Starmap-version of mpp.imap

    Obtained from: https://stackoverflow.com/questions/57354700/
                   starmap-combined-with-tqdm
    """
    if self._state != mpp.RUN:
        raise ValueError("Pool not running")

    if chunksize < 1:
        raise ValueError("Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self._cache)
    self._taskqueue.put((self._guarded_task_generation(result._job,
        mpp.starmapstar, task_batches),
        result._set_length))

    return (item for chunk in result for item in chunk)


def get_subject_dirs(path):
    """Get subject directories from root path

    Args:
        path (str): Path to the directory containing all subject
                    directories

    Returns:
        dir_paths(List[str]): List of subject directory paths
    """
    # Find the directories
    dirs = os.listdir(path)

    # Validate the directories
    dirs = set(filter(lambda x: str.isdigit(x), dirs))

    # Create a list of directory paths
    dir_paths = [path + sd for sd in dirs]

    return dir_paths


mpp.Pool.istarmap = istarmap


