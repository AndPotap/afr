import os
from math import floor
from datetime import datetime
import string
import numpy as np


def add_timestamp(beginning='./params_'):
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_file = beginning + time_stamp
    return output_file


def add_random_characters(beginning, size_to_add=10):
    letters = string.ascii_letters
    stamp = ''
    r = np.random.choice(a=len(letters), size=size_to_add)
    for i in range(size_to_add):
        stamp += letters[r[i]]
    return beginning + stamp


def add_timestamp_with_random(beginning='./params_', ending='.pkl'):
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    time_stamp += '_' + add_random_characters("", size_to_add=4)
    output_file = os.path.join(time_stamp, ending)
    output_file = os.path.join(beginning, output_file)
    return output_file


def print_time_taken(delta, text='Experiment took: ', logger=None):
    minutes = floor(delta / 60)
    seconds = delta - minutes * 60
    message = text + f'{minutes:4d} min and {seconds:4.2f} sec'
    if logger is not None:
        logger.info(message)
    else:
        print(message)
