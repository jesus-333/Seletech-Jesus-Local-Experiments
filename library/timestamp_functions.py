"""
Functions to extract data based on timestamp

@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

import torch
from torch import nn
import numpy as np
import pandas as pd
from datetime import datetime
import sys

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Information from timestamp

def extract_data_from_timestamp(timestamp_entry):
    """
    Given a timestamp in format string YYYY_MM_DD_h_m_s return year, month, day, hour, minutes and seconds as separate int
    """
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

def get_closest_timestamp(timestamp_a : str, timestamps_list : list):
    """
    Given timestamp_a search the closest timestamp in timestamp_list.
    Note that the timestamps must have format YYYY_MM_DD_h_m_s.
    """

    year_1, month_1, day_1, hour_1, minutes_1, seconds_1 = extract_data_from_timestamp(timestamp_a)
    datetime_obj_1 = datetime(year_1, month_1, day_1, hour_1, minutes_1, seconds_1)

    min_difference = sys.maxsize
    idx_closest = 0

    for i in range(len(timestamps_list)):
        year_2, month_2, day_2, hour_2, minutes_2, seconds_2 = extract_data_from_timestamp(timestamps_list[i])

        datetime_obj_1 = datetime(year_1, month_1, day_1, hour_1, minutes_1, seconds_1)
        datetime_obj_2 = datetime(year_2, month_2, day_2, hour_2, minutes_2, seconds_2)

        difference = abs(( datetime_obj_1 - datetime_obj_2 ).total_seconds())

        if difference < min_difference:
            min_difference = difference
            idx_closest = i

    return idx_closest, min_difference

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Select data by timestamp
    
def extract_data_month(data, timestamp, month):
    tmp_index = np.zeros(len(timestamp))
    
    for i in range(len(timestamp)):
      # Extract the day from the timestamp
      tmp_month = int(timestamp[i].split('_')[1])
      
      # If the day correspond to the ones I search, save the position
      if(tmp_month == month): tmp_index[i] = 1
    
    # Extract data and timestamp
    month_data = data[tmp_index == 1, :]
    month_timestamp = timestamp[tmp_index == 1]
    
    return month_data, month_timestamp
    

def extract_data_day(data, timestamp, day):
    tmp_index = np.zeros(len(timestamp))
    
    for i in range(len(timestamp)):
      # Extract the day from the timestamp
      tmp_day = int(timestamp[i].split('_')[2])
      
      # If the day correspond to the ones I search, save the position
      if(tmp_day == day): tmp_index[i] = 1
    
    # Extract data and timestamp
    day_data = data[tmp_index == 1, :]
    day_timestamp = timestamp[tmp_index == 1]
    
    return day_data, day_timestamp


def extract_data_hour(data, timestamp, hour):
    tmp_index = np.zeros(len(timestamp))
    
    for i in range(len(timestamp)):
      # Extract the hour from the timestamp
      tmp_hour = int(timestamp[i].split('_')[3])
      
      # If the hour correspond to the ones I search, save the position
      if(tmp_hour == hour): tmp_index[i] = 1
    
    # Extract data and timestamp
    hour_data = data[tmp_index == 1, :]
    hour_timestamp = timestamp[tmp_index == 1]
    
    return hour_data, hour_timestamp


def extract_data_minute(data, timestamp, minute):
    tmp_index = np.zeros(len(timestamp))
    
    for i in range(len(timestamp)):
      # Extract the hour from the timestamp
      tmp_minute = int(timestamp[i].split('_')[4])
      
      # If the hour correspond to the ones I search, save the position
      if(tmp_minute == minute): tmp_index[i] = 1
    
    # Extract data and timestamp
    minute_data = data[tmp_index == 1, :]
    minute_timestamp = timestamp[tmp_index == 1]
    
    return minute_data, minute_timestamp


def extract_data_month_day(all_data, all_timestamp, month, day):
    month_data, month_timestamp = extract_data_month(all_data, all_timestamp, month)
    day_data, day_timestamp = extract_data_day(month_data, month_timestamp, day)
    
    return day_data, day_timestamp

def extract_data_month_day_hour(all_data, all_timestamp, month, day, hour):
    month_data, month_timestamp = extract_data_month(all_data, all_timestamp, month)
    day_data, day_timestamp = extract_data_day(month_data, month_timestamp, day)
    hour_data, hour_timestamp = extract_data_hour(day_data, day_timestamp, hour)
    
    return hour_data, hour_data

def divide_data_per_day(data, timestamp):
    start_index = 0
    end_index = 0
    actual_day = int(timestamp[0].split('_')[2])
    
    data_per_day = []
    timestamp_per_day = []
    
    
    # Cycle through day
    for i in range(len(timestamp)):
        tmp_day = int(timestamp[i].split('_')[2])
      
        # Check if I arrive in a new day
        if(tmp_day != actual_day):
            end_index = i - 1
        
            # Extract and save data and timestamp for specific day
            tmp_data = data[start_index:end_index, :]
            tmp_timestamp = timestamp[start_index:end_index]
            data_per_day.append(tmp_data)
            timestamp_per_day.append(tmp_timestamp)
        
            # Reset start index and actual_day
            start_index = i
            actual_day = tmp_day
    
    return data_per_day, timestamp_per_day


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Timestamp conversion

def convert_timestamps_format_1(timestamps_list : list):
    """
    Given a list of timestamps in format "YYYY-MM-DD h:m:s" convert them in format "YYYY_MM_DD_h_m_s"
    Used to convert the timestamps of beans to work with previous functions 
    """

    new_timestamps_list = []

    for i in range(len(timestamps_list)):
        old_timestamp = timestamps_list[i]

        date_information, hour_information = old_timestamp.split(" ")

        date_information = date_information.replace('-', '_')
        hour_information = hour_information.replace(':', '_')

        new_timestamp = date_information + '_' + hour_information

        new_timestamps_list.append(new_timestamp)

    return new_timestamps_list


def convert_timestamps_in_dataframe(timestamps_list : list):
    """
    Convert the list of timestamp string in an dataframe of timestamp of dimension len(timestamp_list) x 6
    The 6 columns represents (in order): year, month, day, hour, minute, second
    Each value is saved as a number
    """
    
    timestamp_array = np.zeros((len(timestamps_list), 6))
    
    for i in range(len(timestamps_list)):
        timestamp = timestamps_list[i]
        
        year, month, day, hour, minutes, seconds = extract_data_from_timestamp(timestamp)
        
        timestamp_array[i, 0] = year
        timestamp_array[i, 1] = month
        timestamp_array[i, 2] = day
        timestamp_array[i, 3] = hour
        timestamp_array[i, 4] = minutes
        timestamp_array[i, 5] = seconds
        
    timestamp_dataframe = pd.DataFrame(timestamp_array, columns = ['year', 'month', 'day', 'hour', 'minutes', 'seconds']).astype(int)
    
    return timestamp_dataframe

