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
import torch
from torch.utils.data import DataLoader

from support.datasets import load_spectra_data, load_water_data, create_extended_water_vector, choose_spectra_based_on_water_V1
from support.datasets import PytorchDatasetPlantSpectra_V1
from support.VAE import SpectraVAE_Double_Mems, AttentionVAE
from support.VAE_Conv import SpectraVAE_Double_Mems_Conv

#%% Load data

def load_data_and_create_dataset(config, logger = None):
    """
    Create and return the spectra plants dataset
    
    config is a wandb dictionary that contain all the parameters needed. 
    (see step 2 at http://wandb.me/pytorch-colab)
    """
    
    # Read the spectra data
    spectra_plants_numpy, wavelength, timestamp = load_spectra_data(config.spectra_data_path, config.normalize_trials)
    if logger is not None: logger.debug("Spectra data loaded with normalization = {}".format(config.normalize_trials))
    
    # Read water data and create extend water vector
    water_data, water_timestamp = load_water_data(config.water_data_path)
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

def split_dataset(dataset, config, logger = None):
    if config.percentage_train + config.percentage_test + config.percentage_validation != 1:
        if logger is not None: logger.error("The sum of the percentage of train, test and validation must be 1")
        raise ValueError("The sum of the percentage of train, test and validation must be 1")
        
    dataset_train = dataset[0:int(len(dataset) * config.percentage_train)]
    dateset_test = dataset[int(len(dataset) * config.percentage_train):int(len(dataset) * (config.percentage_train + config.percentage_test))]
    dataset_validation = dataset[int(len(dataset) * (config.percentage_train + config.percentage_test)):]
    if logger is not None: 
        tmp_string = "Dataset split done\n"
        tmp_string += "\t\tLength Training set   = " + str(len(dataset_train)) + "\n"
        tmp_string += "\t\tLength Test set       = " + str(len(dateset_test)) + "\n"
        tmp_string += "\t\tLength Validation set = " + str(len(dataset_validation))
        logger.debug(tmp_string)
    
    if config.print_var:
        print("Length Training set   = " + str(len(dataset_train)))
        print("Length Test set       = " + str(len(dateset_test)))
        print("Length Validation set = " + str(len(dataset_validation)))
        
    return dataset_train, dateset_test, dataset_validation


def make_dataloader(dataset, config, logger = None):
    if 'batch_size' not in config:
        if logger is not None: logger.error("Batch size must be specified and be bigger than 0\n")
        raise ValueError("Batch size must be specified and be bigger than 0")
        
    if 'dataloader_shuffle' not in config: 
        config['dataloader_shuffle'] = True
        if config.print_var: print("Shuffle on dataloader was not specified. It was set to True\n")
        if logger is not None: logger.info("Shuffle on dataloader was not specified. It was set to True")
        
    if 'dataloader_num_worker' not in config: 
        config['dataloader_num_worker'] = 0
        if config.print_var: print("num_worker For dataloader was not specified. It was set to 0\n")
        if logger is not None: logger.info("num_worker For dataloader was not specified. It was set to 0")
    
    loader = DataLoader(dataset, batch_size = config.batch_size,
                                         shuffle = config.dataloader_shuffle,
                                         num_workers = config.dataloader_num_worker)
    
    return loader

#%% Load Model

def get_model_optimizer_scheduler(config, logger = None):
    if config.use_cnn: # Convolutional VAE 
        model = SpectraVAE_Double_Mems_Conv(config.length_mems_1, config.length_mems_2, config.hidden_space_dimension, 
                                            use_as_autoencoder = config.use_as_autoencoder, use_bias = config.use_bias,
                                            print_var = config.print_var)
    else:
        if config.use_attention: # Feed-Forward VAE with attention
            model = AttentionVAE(config.length_mems_1, config.length_mems_2, 
                               config.hidden_space_dimension, config.embedding_size,
                               print_var = config.print_var, use_as_autoencoder = config.use_as_autoencoder )
        else: # Feed-Forward VAE without attention
            model = SpectraVAE_Double_Mems(config.length_mems_1, config.length_mems_2, config.hidden_space_dimension, 
                                         use_as_autoencoder = config.use_as_autoencoder, use_bias = config.use_bias,
                                         print_var = config.print_var)
    
    if 'optimizer_weight_decay' not in config:
        config['optimizer_weight_decay'] = 0.01
        if config.print_var: print("Weight decay for optimizer was not specified. It was set to default value of 0.01\n")
        
    if 'use_scheduler' not in config:
        config['use_scheduler'] = False
        if config.print_var: print("use_scheduler was not specified. It was set to default value of False. So no learning rate scheduler will be used.\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr = config.learning_rate, weight_decay = config.optimizer_weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
    
    return model, optimizer, scheduler