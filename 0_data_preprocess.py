#%% Imports 

import sys
sys.path.insert(0, 'support')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from support.datasets import load_spectra_data, load_water_data, create_extended_water_vector 
from support.timestamp_function import convert_timestamps_in_dataframe

#%% Load data
normalize_trials = 1

spectra_plants_numpy, wavelength, spectra_timestamp = load_spectra_data("data/[2021-08-05_to_11-26]All_PlantSpectra.csv", normalize_trials)

water_data, water_timestamp = load_water_data("data/[2021-08-05_to_11-26]PlantTest_Notes.csv")
extended_water_timestamp = create_extended_water_vector(water_timestamp, water_data, spectra_timestamp)

ht_data = pd.read_csv("data/[2021-08-05_to_11-26]All_PlantHTSensor.csv", encoding= 'unicode_escape')
humidity = ht_data[' Humidity[%]']
temperature = ht_data[' Temperature[C]']
ht_timestamp = ht_data['Timestamp']

#%% Convert timestamp in dataframe

spectra_timestamp_dataframe = convert_timestamps_in_dataframe(spectra_timestamp)
spectra_timestamp_dataframe.to_csv("data/jesus_spectra_timestamp.csv") 

ht_timestamp_dataframe = convert_timestamps_in_dataframe(ht_timestamp)
ht_timestamp_dataframe.to_csv("data/jesus_ht_timestamp.csv") 


#%%
spectra_timestamp_numpy = spectra_timestamp_dataframe.to_numpy()[:, 0:5]
ht_timestamp_numpy = ht_timestamp_dataframe.to_numpy()[:, 0:5]



for i in range(spectra_timestamp_numpy.shape[0]):
    a = spectra_timestamp_numpy[i]
    res = (ht_timestamp_numpy[:, None] == a).all(-1).any(-1)
    
    if(np.sum(res) > 1): print(i, np.sum(res))