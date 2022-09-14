#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 11:39:27 2022

@author: Alberto Zancanaro (jesus)
@organization: University of Padua

This file contain function that read raw data, divide them and create the dataset.
It is created to have cleaner training file and work better with wandb
"""

#%% Imports

from support.datasets import load_spectra_data, load_water_data, create_extended_water_vector, choose_spectra_based_on_water_V1
from support.datasets import PytorchDatasetPlantSpectra_V1

#%%

def load_data_and_create_dataset(config):
    """
    Create and return the spectra plants dataset
    
    config is a wandb dictionary that contain all the parameters needed. 
    (see step 2 at http://wandb.me/pytorch-colab)
    """
    
    if config.percentage_train + config.percentage_test + config.percentage_validation != 1:
        raise ValueError("The sum of the percentage of train, test and validation must be 1")
    
    # Read the spectra data
    spectra_plants_numpy, wavelength, timestamp = load_spectra_data(config.spectra_data_path, config.normalize_trials)
    
    # Read water data and create extend water vector
    water_data, water_timestamp = load_water_data(config.water_data_path)
    extended_water_timestamp = create_extended_water_vector(water_timestamp, water_data, timestamp)

    # Due to the fact that I have much more bad spectra in this way I use them to train the network.
    # Cause I'm lazy I only flip the variable at the beggining and noth change all the variable along the script
    # bad_idx = water, good_idx = NO water
    good_idx, bad_idx = choose_spectra_based_on_water_V1(extended_water_timestamp, 
                                                         time_interval_start = config.time_interval_start, 
                                                         time_interval_end = config.time_interval_end)
    
    good_spectra_dataset = PytorchDatasetPlantSpectra_V1(spectra_plants_numpy[good_idx, :], used_in_cnn = config.use_cnn)
    bad_spectra_dataset = PytorchDatasetPlantSpectra_V1(spectra_plants_numpy[bad_idx, :], used_in_cnn = config.use_cnn)
    
    if config.print_var:
        print("N good spectra = ", len(good_spectra_dataset))
        print("N bad spectra  = ", len(bad_spectra_dataset))
        print("Total spectra  = ", len(good_spectra_dataset) + len(bad_spectra_dataset), "\n")
        
    return good_spectra_dataset, bad_spectra_dataset

