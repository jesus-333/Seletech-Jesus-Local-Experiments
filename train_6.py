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
from support.wandb_init_V2 import build_and_log_model
from support.wandb_init_V2 import build_model, load_dataset, split_dataset
from support.wandb_training_V2 import train_and_log_model
from support.wandb_visualization import bar_loss_wandb_V1

#%% wandb login

wandb.login()

project_name = "Seletech VAE Spectra"

#%% Build Model

model_config = dict(
    length_mems_1 = 300,
    length_mems_2 = 400,
    neurons_per_layer = [32, 64, 36],
    hidden_space_dimension = 4,
    use_as_autoencoder = True,
    use_bias = False,
    use_cnn = False,
    use_attention = False,
    print_var = True
)

# model = build_model(model_config)[0]
build_and_log_model(project_name, model_config)

#%% Load dataset

dataset_config = dict(
    normalize_trials = 1,
    time_interval_start = 45,
    time_interval_end = 360,
    split_percentage_list = [0.7, 0.15, 0.15],
    use_cnn = model_config['use_cnn'],
    print_var = True,
)

good_dataset, bad_dataset = load_dataset(dataset_config)

bad_dataset_train, bad_dataset_test, bad_dataset_validation = split_dataset(bad_dataset, dataset_config)

#%% Train model

training_config = dict(
    model_artifact_name = 'SpectraVAE_FC',
    version = 'v1',
    batch_size = 32,
    lr = 1e-3,
    epochs = 50,
    use_scheduler = True,
    gamma = 0.75, # Parameter of the lr exponential scheduler
    optimizer_weight_decay = 1e-3,
    alpha = 1, # Hyperparameter recon loss
    beta = 3, # Hyperparmeter KL loss
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    log_freq = 1,
    print_var = True,
    dataset_config = dataset_config
)

# Create dataloader 
train_loader = make_dataloader(bad_dataset_train, training_config)
validation_loader = make_dataloader(bad_dataset_validation, training_config)
anomaly_loader = make_dataloader(good_dataset, training_config)
loader_list =[train_loader, validation_loader, anomaly_loader]

# Train model
model = train_and_log_model(project_name, loader_list, training_config)

#%% Error bar plot trained model

plot_config = dict(
    artifact_name = 'SpectraVAE_FC_trained',
    version = 'latest',
    model_name = 'model.pth',
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    figsize = (15, 10),
    dataset_labels = ['Dry (Train)', 'Dry (Validation)', 'Water'],
    ylabel = 'Error',
    colors = ['red', 'orange', 'blue'],
    fontsize = 16
)

train_loader = make_dataloader(bad_dataset_train, training_config)
validation_loader = make_dataloader(bad_dataset_validation, training_config)
anomaly_loader = make_dataloader(good_dataset, training_config)
loader_list =[train_loader, validation_loader, anomaly_loader]

dataloader_list = [train_loader, validation_loader, anomaly_loader]
fig, ax = bar_loss_wandb_V1(project_name, dataloader_list, plot_config)