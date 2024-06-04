"""
@author: Alberto Zancanaro (jesus)
@organization: University of Padua

File used when the repository is cloned from github to colab for training.
The file contain some function to login on google drive and download the data.
"""
#%%

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials# Authenticate and create the PyDrive client.

import pandas as pd
import numpy as np

from support.datasets import spectra_normalization, create_extended_water_vector, choose_spectra_based_on_water_V1
from support.datasets import PytorchDatasetPlantSpectra_V1

#%% Google related function

def google_login():
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)
    
    return drive

def read_data_from_link(drive, link, header = 0):
  id = link.split('/')[5]

  downloaded = drive.CreateFile({'id':id})
  downloaded.GetContentFile(downloaded['title'])  
  df = pd.read_csv(downloaded['title'], header = header, encoding= 'unicode_escape')

  return df

#%% Load data function

def load_data_and_create_dataset_colab(config, drive, logger = None):
    """
    Create and return the spectra plants dataset
    
    config is a wandb dictionary that contain all the parameters needed. 
    (see step 2 at http://wandb.me/pytorch-colab)
    """
    
    # Read the spectra data
    spectra_plants_numpy, wavelength, timestamp = load_spectra_data_colab(drive, config.normalize_trials)
    if logger is not None: logger.debug("Spectra data loaded with normalization = {}".format(config.normalize_trials))
    
    # Read water data and create extend water vector
    water_data, water_timestamp = load_water_data_colab(drive)
    extended_water_timestamp = create_extended_water_vector(water_timestamp, water_data, timestamp)
    if logger is not None: logger.debug("Water data loaded and extended to spectra timestamp")

    # Due to the fact that I have much more bad spectra I use them to train the network.
    good_idx, bad_idx = choose_spectra_based_on_water_V1(extended_water_timestamp, 
                                                         time_interval_start = config.time_interval_start, 
                                                         time_interval_end = config.time_interval_end)
    if logger is not None: logger.debug("Spectra divided in two class based on water quantity")
    
    good_spectra_dataset = PytorchDatasetPlantSpectra_V1(spectra_plants_numpy[good_idx, :], used_in_cnn = config.use_cnn)
    bad_spectra_dataset = PytorchDatasetPlantSpectra_V1(spectra_plants_numpy[bad_idx, :], used_in_cnn = config.use_cnn)
    if logger is not None: 
        tmp_string = "PyTorch dataset created\n"
        tmp_string += "\t\tN good spectra = " + str(len(good_spectra_dataset)) + "\n"
        tmp_string += "\t\tN bad spectra  = " + str(len(bad_spectra_dataset)) + "\n"
        tmp_string += "\t\tTotal spectra  = " + str(len(good_spectra_dataset) + len(bad_spectra_dataset))
        logger.debug(tmp_string)
    
    if config.print_var:
        print("N good spectra = ", len(good_spectra_dataset))
        print("N bad spectra  = ", len(bad_spectra_dataset))
        print("Total spectra  = ", len(good_spectra_dataset) + len(bad_spectra_dataset), "\n")
        
    return good_spectra_dataset, bad_spectra_dataset

def load_spectra_data_colab(drive, normalization_type = -1, print_var = True):
    link = 'https://drive.google.com/file/d/13Prn-VuunlwESDlxQZTR8LRIo741qgDc/view?usp=sharing'
    spectra_plants_df = read_data_from_link(drive, link, header = 2)
    
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

def load_water_data_colab(drive, print_var = True):
    link = 'https://drive.google.com/file/d/14_G50WEC9NNGaJDP-4X9LS0cE5PQR9JU/view?usp=sharing'
    
    log_file_df = read_data_from_link(drive, link)
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
