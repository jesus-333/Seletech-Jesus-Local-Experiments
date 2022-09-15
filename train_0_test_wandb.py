"""
Created on Wed Sep 14 11:15:01 2022

@author: jesus
"""

#%% Imports

import sys
sys.path.insert(1, 'support')

import wandb

from logger import AnotherLogger
from support.initialization import load_data_and_create_dataset, split_dataset, make_dataloader, get_model_optimizer_scheduler
from support.training_wandb import train_model_wandb, save_model_pytorch

#%% Wandb login and log file inizialization

wandb.login()


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
    percentage_validation = 0.2,
    # Model parameters
    hidden_space_dimension = 2,
    use_as_autoencoder = True,
    use_cnn = False,
    use_attention = False,
    embedding_size = 64,
    use_bias = False,
    # Training parameters
    batch_size = 75,
    epochs = 5,
    learning_rate = 1e-3,
    alpha = 1,
    beta = 3,
    optimizer_weight_decay = 1e-3,
    use_scheduler = True,
    gamma = 0.9,
    # Other parameters
    print_var = True,
    training_type = "double mnist water single level",
    log_freq = 1
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
    
    bad_spectra_train_dataloader = make_dataloader(bad_spectra_train, config)
    bad_spectra_test_dataloader = make_dataloader(bad_spectra_test, config)
    bad_spectra_validation_dataloader = make_dataloader(bad_spectra_validaton, config)
    good_dataloader = make_dataloader(good_dataset, config)
    
    vae, optimizer, lr_scheduler = get_model_optimizer_scheduler(config)
    
    train_model_wandb(vae, optimizer,
                      bad_spectra_train_dataloader, bad_spectra_validation_dataloader, good_dataloader, 
                      config, lr_scheduler)
    
    save_model_pytorch(vae, "model.h5")
    