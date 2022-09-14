"""
Created on Wed Sep 14 11:15:01 2022

@author: jesus
"""

#%% Imports

import sys
sys.path.insert(1, 'support')

import numpy as np

import os
import wandb

from logger import AnotherLogger
from support.initialization import load_data_and_create_dataset, split_dataset
# from support.training_wandb import * 

#%% Wandb login and log file inizialization

wandb.login()
os.environ["WANDB_SILENT"] = "true"

logger = AnotherLogger("log.txt")
logger.debug("Wandb login")

#%% Define settings

settings = dict(
    # Data parameters
    spectra_data_path = "data/[2021-08-05_to_11-26]All_PlantSpectra.csv",
    water_data_path = "data/[2021-08-05_to_11-26]PlantTest_Notes.csv",
    labels = ["NO Water (Train)", "NO Water (Test)", "Water"],
    normalize_trials = 1,
    time_interval_start = 45,
    time_interval_end = 360,
    length_mems_1 = 300,
    length_mems_2 = 400,
    percentage_train = 0.6,
    percentage_test = 0.2,
    percentage_validation = 0.9,
    # Model parameters
    hidden_space_dimension = 2,
    use_cnn = False,
    use_attention = False,
    embedding_size = 64,
    # Training parameters
    batch_size = 75,
    epochs = 100,
    learning_rate = 1e-3,
    alpha = 1,
    beta = 3,
    # Other parameters
    print_var = True,
    training_type = "double mnist water single level"
    )

logger.debug("Hyperparameter created")

#%% 

# Tell wandb to get started
with wandb.init(project="test_spectra_wandb", config = settings):
    # Access all HPs through wandb.config, so logging matches execution!
    config = wandb.config
    
    # Load dataset
    good_dataset, bad_dataset = load_data_and_create_dataset(config, logger)
    
    # Split in training, test and validation
    bad_spectra_train, bad_spectra_test, bad_spectra_validaton = split_dataset(bad_dataset, config, logger)
    
    # make the model, data, and optimization problem
    # model, train_loader, test_loader, criterion, optimizer = make(config)
    # print(model)
    
    # # and use them to train the model
    # train(model, train_loader, criterion, optimizer, config)
    
    # # and test its final performance
    # test(model, test_loader)
