"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

This file contains:
    - Function to load the raw data
    - Definition of the PyTorch dataset class for the plant spectra
        
"""

import numpy as np
import pandas as pd

import torch
from torch import nn

from support.timestamp_function import extract_data_from_timestamp, compare_timestamp
from support.preprocess import divide_spectra_in_sequence

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
                    extended_water_vector[i] = round(water_quantity/50)
              
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


class SpectraSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, spectra, config, info_array = None):
        
        # Create spectra sequence and average over sequence period of the info array
        tmp_out = divide_spectra_in_sequence(spectra, config['sequence_length'], config['shift'], info_array)
        
        if info_array is not None:
            spectra_sequence, info_avg = tmp_out[0], tmp_out[1]
        else:
            spectra_sequence = tmp_out
        
        # Convert spectra sequence in Tensor
        self.spectra_sequence = torch.FloatTensor(spectra_sequence[0:-2])
        
        if info_array is not None: self.info_avg = torch.FloatTensor(info_avg)
        else: self.info_avg = None
        
        # Compute label
        if info_array is not None:
            info_avg = info_avg[0:-2]
            wet_idx = info_avg >= np.mean(info_avg) + np.std(info_avg) * config['n_std']
            dry_idx = info_avg <= np.mean(info_avg) - np.std(info_avg) * config['n_std']
            normal_idx = np.logical_and(np.logical_not(wet_idx), np.logical_not(dry_idx))
            if config['binary_label']:
                self.label[normal_idx] = 0
                self.label[np.logical_not(normal_idx)] = 1
            else:
                self.label[normal_idx] = 0
                self.label[wet_idx] = 1
                self.label[dry_idx] = 2
            
        print("len(info_avg[wet_idx]):    ", len(info_avg[wet_idx]))
        print("len(info_avg[dry_idx]):    ", len(info_avg[dry_idx]))
        print("len(info_avg[normal_idx]): ", len(info_avg[normal_idx]))
            
    def __getitem__(self, idx):
        if self.info_avg is None:
            return self.spectra_sequence[idx, :]
        else:
            return self.spectra_sequence[idx, :], self.label[idx].long(), self.info_avg[idx]
    
    def __len__(self):
        return self.label.shape[0]
    

def minutes_to_hour(minutes):
  hour = minutes // 60
  remaining_minutes = minutes % 60

  return hour, remaining_minutes