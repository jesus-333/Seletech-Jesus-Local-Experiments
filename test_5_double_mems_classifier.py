#%%  Imports

import sys
sys.path.insert(1, 'support')

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from support.datasets import load_spectra_data, load_water_data, create_extended_water_vector, choose_spectra_based_on_water_V2
from support.datasets import PytorchDatasetPlantSpectra_V2
from support.full_model import SpectraFramework_FC

#%% Parameters

minute_windows = 8 * 60
minute_shift = 15

normalize_trials = 1

hidden_space_dimension = 2
batch_size = 75
epochs = 200
learning_rate = 1e-3
alpha = 1 # Hyperparemeter to fine tuning the value of the reconstruction error
beta = 2 # Hyperparemeter to fine tuning the value of the KL Loss

print_var = True
step_show = 2

use_as_autoencoder = True

use_cnn = False


#%% Load data

# Sepctra
spectra_plants_numpy, wavelength, timestamp = load_spectra_data("data/[2021-08-05_to_11-26]All_PlantSpectra.csv", normalize_trials)

# Water
water_data, water_timestamp = load_water_data("data/[2021-08-05_to_11-26]PlantTest_Notes.csv")
extended_water_timestamp = create_extended_water_vector(water_timestamp, water_data, timestamp)

# Create average spectra and relative count of water
avg_spectra_matrix, count_water = choose_spectra_based_on_water_V2(spectra_plants_numpy, extended_water_timestamp, minute_windows = minute_windows,  minute_shift = minute_shift)

if(print_var):
    print("Length (in minutes) of the windows: ", minute_windows)
    print("Shift (in minutes) of the windows:  ", minute_shift)
    
    print("Counts of spectra:")
    for label in set(count_water):
        print("\tN. Spectra: {}\t Water Count:{}".format(len(count_water[count_water == label]), label))
        
        
#%% Dataset Creation 

length_mems_1 = int(1650 - min(wavelength))
length_mems_2 = int(max(wavelength) - 1750)

# Create dataset
spectra_dataset = PytorchDatasetPlantSpectra_V2(avg_spectra_matrix, count_water, used_in_cnn = use_cnn, length_mems_1 = length_mems_1, length_mems_2 = length_mems_2)

# Create dataloader
spectra_dataloader = DataLoader(spectra_dataset, batch_size = batch_size, shuffle = True)

# Remove unnecessary variables to optimize memory
del spectra_plants_numpy, avg_spectra_matrix, count_water, timestamp