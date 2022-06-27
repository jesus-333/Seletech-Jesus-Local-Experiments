#%% Imports 

import sys
sys.path.insert(0, 'support')

import numpy as np
import matplotlib.pyplot as plt

from support.datasets import load_spectra_data, load_water_data, create_extended_water_vector 

#%% Load data
normalize_trials = 1

spectra_plants_numpy, wavelength, timestamp = load_spectra_data("data/[2021-08-05_to_11-26]All_PlantSpectra.csv", normalize_trials)

water_data, water_timestamp = load_water_data("data/[2021-08-05_to_11-26]PlantTest_Notes.csv")
extended_water_timestamp = create_extended_water_vector(water_timestamp, water_data, timestamp)

#%%

# N.b. The NIRS sensor record ~1 spectra per minute
hour_windows = 12
minute_windows = hour_windows * 60

minute_shift = 10

avg_spectra_matrix = []
count_water = []

length_mems_1 = int(1650 - min(wavelength))
length_mems_2 = int(max(wavelength) - 1750)

minute = 0
while(True):
    tmp_full_spectra_batch = spectra_plants_numpy[minute:minute + minute_windows, :]
    tmp_timestamp_batch = timestamp[minute:minute + minute_windows]
    count_water.append(np.sum(extended_water_timestamp[minute:minute + minute_windows]))
    
    avg_spectra = np.mean(tmp_full_spectra_batch, 0)
    avg_spectra_matrix.append(avg_spectra)
    
    minute += minute_shift
    if(minute + minute_windows >= spectra_plants_numpy.shape[0]): break


avg_spectra_matrix = np.asarray(avg_spectra_matrix)
count_water = np.asarray(count_water).astype(int)
for n in np.unique(count_water):
    print(n, len(count_water[count_water == n]))