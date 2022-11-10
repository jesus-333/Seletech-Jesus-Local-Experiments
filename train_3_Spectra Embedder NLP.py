"""
Created on Tue Nov  8 16:00:42 2022

@author: Alberto Zancanaro (jesus)
@organization: University of Padua

Train script for NLP spectra embedder
"""

#%% Imports

import sys
sys.path.insert(0, 'support')

import numpy as np
import torch
import wandb

from support.wandb_init_V2 import build_and_log_spectra_embedder_NLP
from support.wandb_training_Spectra_Embedder import train_and_log_SE_model

#%% Wandb login and log file inizialization

wandb.login()

project_name = "Seletech VAE Spectra"

#%% Build model

model_config = dict(
    input_size = 700,
    embedding_size = 2,
    type_embedder= 'skipGram',
    window_size = 2,
    debug = True
    )

build_and_log_spectra_embedder_NLP(project_name, model_config)

#%% Train model

dataset_config = dict(
    # Artifacts info
    artifact_name = 'jesus_333/Seletech VAE Spectra/Dataset_Spectra_1',
    version = 'latest',
    return_other_sensor_data = False,
    spectra_file_name = '[2021-08-05_to_11-26]All_PlantSpectra.csv',
    water_file_name = '[2021-08-05_to_11-26]PlantTest_Notes.csv',
    ht_file_path = '[2021-08-05_to_11-26]All_PlantHTSensor.csv',
    ht_timestamp_path = 'jesus_ht_timestamp.csv', 
    spectra_timstamp_path = 'jesus_spectra_timestamp.csv',
    # Normalization settings
    normalize_trials = 2,
    # Dataset config
    window_size = 2
)

train_config = dict(
    model_artifact_name = 'SpectraEmbedder_skipGram',
    version = 'latest', # REMEMBER ALWAYS TO CHECK THE VERSION
    # Numerical Hyperparameter
    batch_size = 32,
    lr = 1e-1,
    epochs = 20,
    use_scheduler = True,
    gamma = 0.9, # Parameter of the lr exponential scheduler
    optimizer_weight_decay = 1e-2,
    # Support stuff (device, log frequency etc)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    log_freq = 5,
    epoch_to_save_model = 1,
    dataset_config = dataset_config,
    print_var = True,
    debug = True,
)

trained_model = train_and_log_SE_model(project_name, train_config)

#%% End file 