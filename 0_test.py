#%% Imports 

import sys
sys.path.insert(0, 'support')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
import time

from support.datasets import load_spectra_data, load_water_data, create_extended_water_vector , choose_spectra_based_on_water_V1
from support.timestamp_function import convert_timestamps_in_dataframe
from support.preprocess import aggregate_HT_data_V1, aggregate_HT_data_V2

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




