# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 18:11:06 2022

@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)
"""

import numpy as np
import pandas as pd

from support.timestamp_function import extract_data_from_timestamp, extract_spectra_month, extract_spectra_day, extract_spectra_hour, extract_spectra_minute

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
    
    for i in range(spectra_timestamp.shape[0]):
        