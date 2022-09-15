"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

This file contains:
    - Function to load the raw data
    - Function to divide the raw data based on various criteria
    - Definition of the PyTorch dataset class for the plant spectra
        
"""

import numpy as np
import pandas as pd

import torch
from torch import nn

from support.timestamp_function import extract_data_from_timestamp, compare_timestamp

#%% Spectra related function

def load_spectra_data(filename, normalization_type = -1, print_var = True):
    spectra_plants_df = pd.read_csv(filename, header = 2, encoding= 'unicode_escape')
    
    # Convert spectra dataframe in a numpy matrix
    spectra_plants_numpy = spectra_plants_df.iloc[:,5:-1].to_numpy(dtype = float)
    if(print_var): print("Spectra matrix shape\t=\t", spectra_plants_numpy.shape, "\t (Time x Wavelength)")
    
    # Recover wavelength
    wavelength = spectra_plants_df.keys()[5:-1].to_numpy(dtype = float)
    if(print_var): print("Wavelength vector shape\t=\t", wavelength.shape)
    
    # Recover timestamp
    timestamp = spectra_plants_df['# Timestamp'].to_numpy()
    if(print_var): print("Timestamp vector shape\t=\t", timestamp.shape)
    
    if(normalization_type == 0 or normalization_type == 1): 
        spectra_plants_numpy = spectra_normalization(spectra_plants_numpy, normalization_type)
    
    return spectra_plants_numpy, wavelength, timestamp


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
    return spectra / compute_normalization_factor(spectra, norm_type)


#%% Water related information

def load_water_data(filename, print_var = True):
    log_file_df =  pd.read_csv(filename, encoding= 'unicode_escape')
    if print_var: print("Water File Loaded\n")
    
    # Extract timestamp (in the same format of the spectra file)
    tmp_data = log_file_df['Date']
    tmp_hour = log_file_df['Time']
    log_timestamp = []
    for data, hour in zip(tmp_data, tmp_hour):
      tmp_timestamp = data.split('/')[2] + '_' + data.split('/')[1] + '_' + data.split('/')[0] + '_'
      tmp_timestamp += hour.split(':')[0] + '_' + hour.split(':')[1] + '_00'
    
      log_timestamp.append(tmp_timestamp)
    
    # Extract daily water
    water = np.asarray(log_file_df['H2O[g]'])
    water[np.isnan(water)] = 0
    
    log_timestamp = np.asarray(log_timestamp)[2:]
    water = water[2:]
    
    if print_var: print("Water vector length =\t" , len(water))
    if print_var: print("Log timestamp length =\t" , len(log_timestamp), "\n")
    
    return water, log_timestamp


def create_extended_water_vector(water_log_timestamp, water_vector, spectra_timestamp):
    """
    Create a vector of index that indicate in the spectra timestamp the closest to when water was given to the plant
    """

    j = 0
    actual_water_timestamp = water_log_timestamp[j]
    water_quantity = water_vector[j]
    extended_water_vector = np.zeros(len(spectra_timestamp))
    w_year, w_month, w_day, w_hour, w_minutes, w_seconds = extract_data_from_timestamp(actual_water_timestamp)
    
    for i in range(len(spectra_timestamp)):
        actual_spectra_timestamp = spectra_timestamp[i]
          
        sp_year, sp_month, sp_day, sp_hour, sp_minutes, sp_seconds = extract_data_from_timestamp(actual_spectra_timestamp)
          
        if(sp_year == w_year and sp_month == w_month and sp_day == w_day):
            if(sp_hour >= w_hour and sp_minutes >= w_minutes):
                if(water_quantity > 0): 
                    extended_water_vector[i] = 1
              
                j += 1
                actual_water_timestamp = water_log_timestamp[j]
                water_quantity = water_vector[j]
                w_year, w_month, w_day, w_hour, w_minutes, w_seconds = extract_data_from_timestamp(actual_water_timestamp)
    
        if(compare_timestamp(actual_spectra_timestamp, actual_water_timestamp)):
            j += 1
            actual_water_timestamp = water_log_timestamp[j]
            water_quantity = water_vector[j]
            w_year, w_month, w_day, w_hour, w_minutes, w_seconds = extract_data_from_timestamp(actual_water_timestamp)
    

    return extended_water_vector


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

    good_timestamp = np.where(extended_water_timestamp == 1)[0]

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
   

#%% Dataset declaration

class PytorchDatasetPlantSpectra_V1(torch.utils.data.Dataset):
    """
    Extension of PyTorch Dataset class to work with spectra plants data. It DON'T DIVIDE the spectra in mems1 and mems2
    """
    
    # Inizialization method
    def __init__(self, spectra_data, used_in_cnn = False):
        self.spectra = torch.from_numpy(spectra_data).float()
        
        if(used_in_cnn): self.spectra = self.spectra.unsqueeze(1)
      
    def __getitem__(self, idx):
        return self.spectra[idx, :]
    
    def __len__(self):
        return self.spectra.shape[0]
    

class PytorchDatasetPlantSpectra_V2(torch.utils.data.Dataset):
    """
    Extension of PyTorch Dataset class to work with spectra plants data. Used for VAE + Classifier
    It CAN DIVIDE the spectra in mems1 and mems2 
    """
    
    # Inizialization method
    def __init__(self, spectra_data, count_water, used_in_cnn = False, length_mems_1 = -1, length_mems_2 = -1):
        self.spectra = torch.from_numpy(spectra_data).float()
        self.label = torch.from_numpy(count_water).int()
        
        if(used_in_cnn): self.spectra = self.spectra.unsqueeze(1)
        
        if(length_mems_1 > 0 and length_mems_2 > 0):
            self.divide_spectra = True
            self.length_mems_1 = length_mems_1
            self.length_mems_2 = length_mems_2
        else:
            self.divide_spectra = False
      
    def __getitem__(self, idx):
        if(self.divide_spectra):
            tmp_spectra = self.spectra[idx, :]
            spectra_mems_1 = tmp_spectra[..., 0:self.length_mems_1]
            spectra_mems_2 = tmp_spectra[..., (- 1 - self.length_mems_2):-1]
            
            return spectra_mems_1, spectra_mems_2, self.label[idx]
        else:   
            return self.spectra[idx, :], self.label[idx]
    
    def __len__(self):
        return self.spectra.shape[0]



def minutes_to_hour(minutes):
  hour = minutes // 60
  remaining_minutes = minutes % 60

  return hour, remaining_minutes