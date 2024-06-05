# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 18:11:06 2022

@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

This file contain function related to extra data (e.g. humidity, water).
It also contains function to divided the spectra in sequence

"""

import numpy as np

from support.timestamp_function import extract_data_from_timestamp, extract_spectra_month, extract_spectra_day, extract_spectra_hour, extract_spectra_minute

#%% Spectra Normalization

def compute_normalization_factor(spectra, norm_type):
    if(norm_type == 0): # Half normalization
        tmp_sum = np.sum(spectra, 0)
        normalization_factor = tmp_sum / spectra.shape[0]
    elif(norm_type == 1): # Full normalization
        tmp_sum = np.sum(spectra)
        normalization_factor = tmp_sum / (spectra.shape[0] * spectra.shape[1])
    else: 
        normalization_factor = 0
    
    return normalization_factor

def spectra_normalization(spectra, norm_type):
    if norm_type == 0 : # Seletech 1 - Half normalization
        tmp_sum = np.sum(spectra, 0)
        normalization_factor = tmp_sum / spectra.shape[0]
        normalized_spectra = spectra / normalization_factor
    elif norm_type == 1 : # Seletech 2 - Full normalization
        tmp_sum = np.sum(spectra)
        normalization_factor = tmp_sum / (spectra.shape[0] * spectra.shape[1])
        normalized_spectra = spectra / normalization_factor 
    elif norm_type == 2: # Absorbance
        normalized_spectra = np.log10(1 / spectra)
    else: 
        raise ValueError("Normalization type must be 0,1 or 2")
    
    return normalized_spectra
   

#%% Humidity and temperature data

def aggregate_HT_data_V1(ht_timestamp, spectra_timestamp, h_array, t_array):
    """
    The HT sensor have a much higher sampling frequency (~1s) respect the mems (~1 minute)
    This function aggregate the data of an entire minute of the HT sensor to have (circa) 1 measurement per minutes and 1 by 1 correspondence with the spectra data
    Used timestamp in the original format (string)
    """
    
    aggregate_h_array = []
    aggregate_t_array = []
    aggregate_timestamp = []
    
    prev_month, prev_day, prev_hour = -1, -1, -1
    
    for i in range(len(spectra_timestamp)):
        # if(i % 3 == 0): print("Completition: {}".format(round(i/len(spectra_timestamp) * 100, 6)))
        actual_spectra_timestamp = spectra_timestamp[i]
      
        year, month, day, hour, minutes, seconds = extract_data_from_timestamp(actual_spectra_timestamp)
      
        if(month != prev_month):
            prev_month = month
            tmp_month_h_data, tmp_month_h_timestamp = extract_spectra_month(h_array, ht_timestamp, month)
            tmp_month_t_data, tmp_month_t_timestamp = extract_spectra_month(t_array, ht_timestamp, month)
      
        if(day != prev_day):
            prev_day = day
            tmp_day_h_data, tmp_day_h_timestamp = extract_spectra_day(tmp_month_h_data, tmp_month_h_timestamp, day)
            tmp_day_t_data, tmp_day_t_timestamp = extract_spectra_day(tmp_month_t_data, tmp_month_t_timestamp, day)
      
        if(hour != prev_hour):
            prev_hour = hour
            tmp_hour_h_data, tmp_hour_h_timestamp = extract_spectra_hour(tmp_day_h_data, tmp_day_h_timestamp, hour)
            tmp_hour_t_data, tmp_hour_t_timestamp = extract_spectra_hour(tmp_day_t_data, tmp_day_t_timestamp, hour)
      
        tmp_h_data_minute, tmp_h_timestamp_minute = extract_spectra_minute(tmp_hour_h_data, tmp_hour_h_timestamp, minutes)
        if(len(tmp_h_data_minute) > 0):
            aggregate_h_array.append(np.mean(tmp_h_data_minute))
        else:
            if(len(aggregate_h_array) > 0): aggregate_h_array.append(aggregate_h_array[-1])
            else: aggregate_h_array.append(tmp_hour_h_data[0])
      
        tmp_t_data_minute, tmp_t_timestamp_minute = extract_spectra_minute(tmp_hour_t_data, tmp_hour_t_timestamp, minutes)
        if(len(tmp_t_data_minute) > 0):
            aggregate_t_array.append(np.mean(tmp_t_data_minute))
        else:
            if(len(aggregate_t_array) > 0): aggregate_t_array.append(aggregate_t_array[-1])
            else: aggregate_t_array.append(tmp_hour_t_data[0])
      
        aggregate_timestamp.append('{}_{}_{}_{}_{}_{}'.format(year, month, day, hour, minutes, seconds))
    
    return aggregate_h_array, aggregate_t_array, aggregate_timestamp


def aggregate_HT_data_V2(ht_timestamp, spectra_timestamp, h_array, t_array):  
    """
    The HT sensor have a much higher sampling frequency (~1s) respect the mems (~1 minute)
    This function aggregate the data of an entire minute of the HT sensor to have (circa) 1 measurement per minutes and 1 by 1 correspondence with the spectra data
    Used timestamp in the new formmat (array)
    """
    
    prev_month, prev_day, prev_hour = -1, -1, -1
    
    aggregate_h_array = []
    aggregate_t_array = []
    
    for i in range(spectra_timestamp.shape[0]):
        year, month, day, hour, minute, second = spectra_timestamp[i]
        
        # TODO Add section with year
        
        if(month != prev_month):
            prev_month = month
            
            tmp_idx = ht_timestamp[:, 1] == month
            
            h_data_month = h_array[tmp_idx]
            t_data_month = t_array[tmp_idx]
            ht_timestamp_month = ht_timestamp[tmp_idx]
            
        if(day != prev_day):
            prev_day = day
            
            tmp_idx = ht_timestamp_month[:, 2] == day
            
            h_data_day = h_data_month[tmp_idx]
            t_data_day = t_data_month[tmp_idx]
            ht_timestamp_day = ht_timestamp_month[tmp_idx]
            
        if(hour != prev_hour):
            prev_hour = hour
            
            tmp_idx = ht_timestamp_day[:, 3] == hour
            
            h_data_hour = h_data_day[tmp_idx]
            t_data_hour = t_data_day[tmp_idx]
            ht_timestamp_hour = ht_timestamp_day[tmp_idx]
            
        
        tmp_idx = ht_timestamp_hour[:, 4] == minute
        
        h_data_minute = h_data_hour[tmp_idx]
        t_data_minute = t_data_hour[tmp_idx]
        # ht_timestamp_minute = ht_timestamp_hour[tmp_idx]

        if(len(h_data_minute) > 0):
            aggregate_h_array.append(np.mean(h_data_minute))
        else:
            # aggregate_h_array.append(np.mean(h_data_hour))
            if(len(aggregate_h_array) > 0): aggregate_h_array.append(aggregate_h_array[-1])
            else: aggregate_h_array.append(np.mean(h_data_hour))
      
        if(len(t_data_minute) > 0):
            aggregate_t_array.append(np.mean(t_data_minute))
        else:
            # aggregate_t_array.append(np.mean(t_data_hour))
            if(len(aggregate_t_array) > 0): aggregate_t_array.append(aggregate_t_array[-1])
            else: aggregate_t_array.append(np.mean(t_data_hour))
    
    aggregate_timestamp = np.copy(spectra_timestamp)
    return np.asarray(aggregate_h_array), np.asarray(aggregate_t_array), aggregate_timestamp


#%% Divide spectra with water

def choose_spectra_based_on_water_V1(extended_water_timestamp, time_interval_start, time_interval_end):
    """
    Create an index vector containg all the spectra after the plant was given water.
    The after is defined with the two variable time_interval_start and time_interval_end and is based on the number of samples.
    N.b. time_interval_end > time_interval_start
    
    # TODO
    If the water was given to the plant at hour XX:YY then the taken spectra will be between XX:YY + time_interval_start and XX:YY  +time_interval_end 
    """
    
    if(time_interval_start >= time_interval_end): raise Exception("time_interval_start should be greater than time_interval_end")
    
    good_idx_tmp = np.zeros(len(extended_water_timestamp))

    good_timestamp = np.where(extended_water_timestamp != 0)[0]

    for idx in good_timestamp: good_idx_tmp[idx + time_interval_start:idx + time_interval_end] = 1
    
    good_idx = good_idx_tmp == 1 
    bad_idx = good_idx_tmp != 1 
    
    return good_idx, bad_idx


def choose_spectra_based_on_water_V2(spectra_data, extended_water_timestamp, minute_windows = 8 * 60,  minute_shift = 15):
    """
    Take all the spectra in an interval of time (specified in minutes) and average them. It also count the number of time water was given in that period of time.
    Once done the averaging windows is shifted forward of tot minutes.
    
    Input data: the spectra matrix and the extended_water_timestamp (obtained with the function create_extended_water_vector)
    Parameter: minute_windows (length in minutes of the interval, must be an integer), minute_shift (how many minutes shift the averaging windows forward)
    Output: avg_spectra_matrix (matrix of dimension n x wavelenght, each row is an average of various spectra), count_water (array of length n, each row contains the number of times water was given to the plant for the corresponding spectra)
    """
    avg_spectra_matrix = []
    count_water = []
    
    minute = 0
    while(True):
        tmp_full_spectra_batch = spectra_data[minute:minute + minute_windows, :]
        count_water.append(np.sum(extended_water_timestamp[minute:minute + minute_windows]))
        
        avg_spectra = np.mean(tmp_full_spectra_batch, 0)
        avg_spectra_matrix.append(avg_spectra)
        
        minute += minute_shift
        if(minute + minute_windows >= spectra_data.shape[0]): break
    
    
    avg_spectra_matrix = np.asarray(avg_spectra_matrix)
    count_water = np.asarray(count_water).astype(int)       

    
    return avg_spectra_matrix, count_water  

#%% Sequence function

def divide_spectra_in_sequence(spectra, sequence_length, shift = -1, info_array = None):
    """
    Divide the matrix of spectra in list of spectra. Each list contain sequence_length spectra.
    
    If an info vector is passed it also compute the average of the info vector for each sequence. The info vector must have the same number of element of the spectra matrix
    (The info array are the array with the data from the other sensors)
    """

    if info_array is not None:
        if len(info_array) != spectra.shape[0]: raise ValueError("The info array length and the number of spectra must be equal")
        info_avg = []
        
    sequence_list = []
    
    if shift <= 0: shift = sequence_length
    
    i = 0
    while True:
        if i + sequence_length >= spectra.shape[0]:
            tmp_sequence = spectra[i:-1]
            sequence_list.append(tmp_sequence)
            if info_array is not None: info_avg.append(np.mean(info_array[i:-1]))
            break
        else:
            tmp_sequence = spectra[i:i+sequence_length]
            sequence_list.append(tmp_sequence)
            if info_array is not None: info_avg.append(np.mean(info_array[i:i+sequence_length]))
            
        i += shift
    
    if info_array is not None:
        return sequence_list, np.asarray(info_avg)
    else:       
        return sequence_list    
    
