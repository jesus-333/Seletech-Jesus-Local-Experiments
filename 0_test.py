#%% Imports 

import sys
sys.path.insert(0, 'support')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
import time
import wandb

from support.datasets import load_spectra_data, load_water_data, create_extended_water_vector
from support.timestamp_function import convert_timestamps_in_dataframe
from support.preprocess import aggregate_HT_data_V1, aggregate_HT_data_V2, choose_spectra_based_on_water_V1

#%% Load data
normalize_trials = -1

spectra_plants_numpy, wavelength, spectra_timestamp = load_spectra_data("data/[2021-08-05_to_11-26]All_PlantSpectra.csv", normalize_trials)

water_data, water_timestamp = load_water_data("data/[2021-08-05_to_11-26]PlantTest_Notes.csv")
extended_water_timestamp = create_extended_water_vector(water_timestamp, water_data, spectra_timestamp)

ht_data = pd.read_csv("data/[2021-08-05_to_11-26]All_PlantHTSensor.csv", encoding= 'unicode_escape')
humidity = ht_data[' Humidity[%]']
temperature = ht_data[' Temperature[C]']
ht_timestamp = ht_data['Timestamp']

#%% Choose spectra

time_interval_start = 45
time_interval_end = 360

good_idx, bad_idx = choose_spectra_based_on_water_V1(extended_water_timestamp, time_interval_start = time_interval_start, time_interval_end = time_interval_end)

#%%
import wandb
from support.wandb_init_V2 import load_dataset_from_artifact

wandb.login()
load_config = dict(
    # Info for spectra
    artifact_name = 'jesus_333/Seletech VAE Spectra/Dataset_Spectra_1',
    version = 'latest',
    spectra_file_name = '[2021-08-05_to_11-26]All_PlantSpectra.csv',
    normalize_trials = 1,
    # Info for other sensor data
    return_other_sensor_data = True,
    water_file_name = '[2021-08-05_to_11-26]PlantTest_Notes.csv',
    ht_file_path = '[2021-08-05_to_11-26]All_PlantHTSensor.csv',
    ht_timestamp_path = 'jesus_ht_timestamp.csv', 
    spectra_timstamp_path = 'jesus_spectra_timestamp.csv'
)

a = load_dataset_from_artifact(load_config)

#%%
from support.preprocess import divide_spectra_in_sequence
import torch

spectra = a[0]
h_array = a[4]

sequence_length = 10
shift = int(sequence_length / 2)
spectra_sequence, info_avg = divide_spectra_in_sequence(spectra, sequence_length, shift, h_array)

print(spectra_sequence[0].shape)
print(spectra_sequence[int(len(spectra_sequence)/2)].shape)
print(spectra_sequence[-1].shape)

print(spectra.shape[0] // sequence_length)
print(spectra.shape[0] % sequence_length)

# b = torch.FloatTensor(spectra_sequence[0:-2])

plt.figure(figsize = (15, 10))
plt.plot(np.linspace(0,1, len(h_array)), h_array)
plt.plot(np.linspace(0,1, len(info_avg)), info_avg)