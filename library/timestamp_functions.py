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

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Select spectra by timestamp

def extract_spectra_month_OLD(all_spectra, all_timestamp, month):
    # Small dictionary with the index of the start of each month for the file with all plant spectra
    month_index = {8: 0, 9: 37100, 10: 78818, 11: 121929}
    
    if(month < 8 or month > 11): Exception("ERROR. Month must be between 8 and 11") 
    else:
        start_index = month_index[month]
        end_index = month_index[month + 1] - 1
        
        return  all_spectra[start_index:end_index, :], all_timestamp[start_index:end_index]
    
def extract_spectra_month(spectra, timestamp, month):
    tmp_index = np.zeros(len(timestamp))
    
    for i in range(len(timestamp)):
      # Extract the day from the timestamp
      tmp_month = int(timestamp[i].split('_')[1])
      
      # If the day correspond to the ones I search, save the position
      if(tmp_month == month): tmp_index[i] = 1
    
    # Extract spectra and timestamp
    month_spectra = spectra[tmp_index == 1, :]
    month_timestamp = timestamp[tmp_index == 1]
    
    return month_spectra, month_timestamp
    

def extract_spectra_day(spectra, timestamp, day):
    tmp_index = np.zeros(len(timestamp))
    
    for i in range(len(timestamp)):
      # Extract the day from the timestamp
      tmp_day = int(timestamp[i].split('_')[2])
      
      # If the day correspond to the ones I search, save the position
      if(tmp_day == day): tmp_index[i] = 1
    
    # Extract spectra and timestamp
    day_spectra = spectra[tmp_index == 1, :]
    day_timestamp = timestamp[tmp_index == 1]
    
    return day_spectra, day_timestamp


def extract_spectra_hour(spectra, timestamp, hour):
    tmp_index = np.zeros(len(timestamp))
    
    for i in range(len(timestamp)):
      # Extract the hour from the timestamp
      tmp_hour = int(timestamp[i].split('_')[3])
      
      # If the hour correspond to the ones I search, save the position
      if(tmp_hour == hour): tmp_index[i] = 1
    
    # Extract spectra and timestamp
    hour_spectra = spectra[tmp_index == 1, :]
    hour_timestamp = timestamp[tmp_index == 1]
    
    return hour_spectra, hour_timestamp


def extract_spectra_minute(spectra, timestamp, minute):
    tmp_index = np.zeros(len(timestamp))
    
    for i in range(len(timestamp)):
      # Extract the hour from the timestamp
      tmp_minute = int(timestamp[i].split('_')[4])
      
      # If the hour correspond to the ones I search, save the position
      if(tmp_minute == minute): tmp_index[i] = 1
    
    # Extract spectra and timestamp
    minute_spectra = spectra[tmp_index == 1, :]
    minute_timestamp = timestamp[tmp_index == 1]
    
    return minute_spectra, minute_timestamp


def extract_spectra_month_day(all_spectra, all_timestamp, month, day):
    month_spectra, month_timestamp = extract_spectra_month(all_spectra, all_timestamp, month)
    day_spectra, day_timestamp = extract_spectra_day(month_spectra, month_timestamp, day)
    
    return day_spectra, day_timestamp


def extract_spectra_month_day_hour(all_spectra, all_timestamp, month, day, hour):
    month_spectra, month_timestamp = extract_spectra_month(all_spectra, all_timestamp, month)
    day_spectra, day_timestamp = extract_spectra_day(month_spectra, month_timestamp, day)
    hour_spectra, hour_timestamp = extract_spectra_hour(day_spectra, day_timestamp, hour)
    
    return hour_spectra, hour_spectra


def divide_spectra_per_day(spectra, timestamp):
    start_index = 0
    end_index = 0
    actual_day = int(timestamp[0].split('_')[2])
    
    spectra_per_day = []
    timestamp_per_day = []
    
    
    # Cycle through day
    for i in range(len(timestamp)):
        tmp_day = int(timestamp[i].split('_')[2])
      
        # Check if I arrive in a new day
        if(tmp_day != actual_day):
            end_index = i - 1
        
            # Extract and save spectra and timestamp for specific day
            tmp_spectra = spectra[start_index:end_index, :]
            tmp_timestamp = timestamp[start_index:end_index]
            spectra_per_day.append(tmp_spectra)
            timestamp_per_day.append(tmp_timestamp)
        
            # Reset start index and actual_day
            start_index = i
            actual_day = tmp_day
    
    return spectra_per_day, timestamp_per_day


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
