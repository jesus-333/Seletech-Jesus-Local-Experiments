"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)
"""

#%% Imports

import torch
from torch import nn

#%% Timestamp related function

def extract_data_from_timestamp(timestamp_entry):
    year = int(timestamp_entry.split('_')[0])
    month = int(timestamp_entry.split('_')[1])
    day = int(timestamp_entry.split('_')[2])
    hour = int(timestamp_entry.split('_')[3])
    minutes = int(timestamp_entry.split('_')[4])
    seconds = int(timestamp_entry.split('_')[5])
    
    return year, month, day, hour, minutes, seconds


def compare_timestamp(timestamp_1, timestamp_2):
    """
    Function that return True if and only if timestamp 1 is more recent that tiemstamp 2
    """
    
    year_1, month_1, day_1, hour_1, minutes_1, seconds_1 = extract_data_from_timestamp(timestamp_1)
    year_2, month_2, day_2, hour_2, minutes_2, seconds_2 = extract_data_from_timestamp(timestamp_2)
    
    if(year_1 > year_2): return True
    if(year_1 == year_2 and month_1 > month_2): return True
    if(year_1 == year_2 and month_1 == month_2 and day_1 > day_2): return True
    if(year_1 == year_2 and month_1 == month_2 and day_1 == day_2 and hour_1 > hour_2): return True
    if(year_1 == year_2 and month_1 == month_2 and day_1 == day_2 and hour_1 == hour_2 and minutes_1 > minutes_2): return True
    if(year_1 == year_2 and month_1 == month_2 and day_1 == day_2 and hour_1 == hour_2 and minutes_1 == minutes_2 and seconds_1 > seconds_2): return True
    
    return False