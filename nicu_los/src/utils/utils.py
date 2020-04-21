#!/usr/bin/python3

"""utils.py - Utils for extracting data of newborn patients from the
              MIMIC-III CSVs.

As part of my Master's thesis at KTH Royal Institute of Technology.
"""

__author__ = "Bas Straathof"

import datetime, os, re, shutil, time

import multiprocessing.pool as mpp


def get_date(s_date):
    date_patterns = ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S"]

    for pattern in date_patterns:
        try:
            return datetime.datetime.strptime(s_date, pattern)
        except:
            pass

    print(f'Date is not in expected format: {s_date}')
    sys.exit(0)


def round_up_to_hour(dt):
    """Round datetime up to hour

    Args:
        dt (datetime.datetime/str): Original atetime

    Returns:
        dt (datetime.datetime/str): Datetime rounded up to hour
    """
    if type(dt) == str:
       dt = get_date(dt)

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


