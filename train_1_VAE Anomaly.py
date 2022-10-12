"""
Created on Fri Sep 23 14:43:48 2022

@author: Alberto Zancanaro (jesus)
@organization: University of Padua
"""

#%%

import sys
sys.path.insert(0, 'support')

import torch
import wandb

from support.wandb_init_V1 import make_dataloader
from support.wandb_init_V2 import build_and_log_VAE_model
from support.wandb_init_V2 import build_VAE_model, load_dataset_local_VAE, split_dataset
from support.wandb_training_VAE import train_and_log_VAE_model
from support.wandb_visualization import bar_loss_wandb_V1

#%% wandb login

wandb.login()

project_name = "Seletech VAE Spectra"

#%% Build Model

model_config = dict(
    length_mems_1 = 300,
    length_mems_2 = 400,
    neurons_per_layer = [64, 128, 36],
    hidden_space_dimension = 2,
    use_as_autoencoder = False,
    use_bias = False,
    use_cnn = False,
    use_attention = False,
    print_var = True
)

# model = build_model(model_config)[0]
build_and_log_VAE_model(project_name, model_config)

#%% Load dataset

dataset_config = dict(
    normalize_trials = 1,
    time_interval_start = 45,
    time_interval_end = 360,
    split_percentage_list = [0.7, 0.15, 0.15],
    use_cnn = model_config['use_cnn'],
    print_var = True,
)

good_dataset, bad_dataset = load_dataset_local_VAE(dataset_config)

bad_dataset_train, bad_dataset_test, bad_dataset_validation = split_dataset(bad_dataset, dataset_config)

#%% Train model

"""
v1: AE.  neurons_per_layer = [64, 128, 36]. Hidden_space = 2
v2: AE.  neurons_per_layer = [32, 64, 36].  Hidden_space = 4
v6: VAE. neurons_per_layer = [32, 64, 36].  Hidden_space = 2
v7: VAE. neurons_per_layer = [64, 128, 36]. Hidden_space = 2
"""

training_config = dict(
    model_artifact_name = 'SpectraVAE_FC',
    version = 'v2', # REMEMBER ALWAYS TO CHECK THE VERSION
    batch_size = 32,
    lr = 1e-2,
    epochs = 50,
    use_scheduler = True,
    gamma = 0.9, # Parameter of the lr exponential scheduler
    optimizer_weight_decay = 1e-3,
    alpha = 1, # Hyperparameter recon loss
    beta = 3, # Hyperparmeter KL loss
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    log_freq = 1,
    print_var = True,
)

dataset_config = dict(
    # Artifacts info
    artifact_name = 'Dataset_Spectra_1',
    version = 'latest',
    return_other_sensor_data = True,
    spectra_file_name = '[2021-08-05_to_11-26]All_PlantSpectra.csv',
    water_file_name = '[2021-08-05_to_11-26]PlantTest_Notes.csv',
    ht_file_path = '[2021-08-05_to_11-26]All_PlantHTSensor.csv',
    ht_timestamp_path = 'jesus_ht_timestamp.csv', 
    spectra_timstamp_path = 'jesus_spectra_timestamp.csv',
    # Dataset settings
    normalize_trials = 1,
    time_interval_start = 45,
    time_interval_end = 360,
    split_percentage_list = [0.7, 0.15, 0.15],
    print_var = True,
)

training_config['dataset_config'] = dataset_config

# Train model
model = train_and_log_VAE_model(project_name, training_config)

#%% Error bar plot trained model

plot_config = dict(
    # Model settings
    artifact_name = 'SpectraVAE_FC_trained',
    version = 'latest', # REMEMBER ALWAYS TO CHECK THE VERSION
    model_file_name = 'model',
    epoch_of_model = -1, # Retrieve the model saved at this epoch during training
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    batch_size = 32,
    figsize = (15, 10),
    dataset_labels = ['Dry (Train)', 'Dry (Validation)', 'Water'],
    ylabel = 'Error',
    colors = ['red', 'orange', 'skyblue'],
    fontsize = 16,
    add_std_bar = True,
    error_kw = dict(ecolor = 'black', lw = 5, capsize = 15, capthick = 5)
)

dataset_config = dict(
    # Artifacts info
    artifact_name = 'Dataset_Spectra_1',
    version = 'latest',
    return_other_sensor_data = False,
    spectra_file_name = '[2021-08-05_to_11-26]All_PlantSpectra.csv',
    # Dataset settings
    normalize_trials = 1,
)

plot_config['dataset_config'] = dataset_config

fig, ax = bar_loss_wandb_V1(project_name, plot_config)
