# Imports 

import sys
sys.path.insert(1, 'support')

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from support.datasets import load_spectra_data, load_water_data, create_extended_water_vector, choose_spectra_based_on_water 
from support.datasets import PytorchDatasetPlantSpectra_V1
from support.VAE import SpectraVAE_Double_Mems
from support.VAE_Conv import SpectraVAE_Double_Mems_Conv
from support.training import advanceEpochV2, VAE_loss, advance_recon_loss, advanceEpochV3
from support.visualization import compare_results_by_spectra, compare_results_by_loss, draw_hist_loss
from support.visualization import visualize_latent_space_V1, visualize_latent_space_V2, visualize_latent_space_V3, visualize_latent_space_V4

#%% Parameters

# min_threeshold = 0.64 # For lower
# min_threeshold = 0.5 # For upper
normalize_trials = 1
percentage_train = 0.77

hidden_space_dimension = 2
batch_size = 77
epochs = 60
learning_rate = 1e-3 / 2
alpha = 1 # Hyperparemeter to fine tuning the value of the reconstruction error
beta = 3 # Hyperparemeter to fine tuning the value of the KL Loss

print_var = True
step_show = 2

use_as_autoencoder = True

#%% Load data

# Variable used in dataset creation. Add an extra dimension for the channels
used_in_cnn = True

# N.b. If use as normal autoencoder the dimension of the hidden space is doubled for construction reasons. 
# So I divided it by 2 to have the one that I specified in the settings
# This work well only to even hidden space dimension
if(use_as_autoencoder): hidden_space_dimension = int(hidden_space_dimension/2)

# Sepctra
spectra_plants_numpy, wavelength, timestamp = load_spectra_data("data/[2021-08-05_to_11-26]All_PlantSpectra.csv", normalize_trials)

# Water
water_data, water_timestamp = load_water_data("data/[2021-08-05_to_11-26]PlantTest_Notes.csv")
extended_water_timestamp = create_extended_water_vector(water_timestamp, water_data, timestamp)

full_spectra_dataset = PytorchDatasetPlantSpectra_V1(spectra_plants_numpy, used_in_cnn = used_in_cnn)

# Select randomly a percentage of indices
idx = np.linspace(0, len(full_spectra_dataset) - 1, len(full_spectra_dataset), dtype = int)
tmp_idx = np.random.choice(idx, int(len(full_spectra_dataset) * percentage_train), replace = False).astype(int)

# Set that index to True for training set
train_idx = np.zeros(len(full_spectra_dataset), dtype = bool)
train_idx[tmp_idx] = True
train_dataloader = DataLoader(full_spectra_dataset[train_idx], batch_size = batch_size, shuffle = True)

# Set that index to False for training set
test_idx = np.ones(len(full_spectra_dataset), dtype = bool)
test_idx[tmp_idx] = False
test_dataloader = DataLoader(full_spectra_dataset[test_idx], batch_size = batch_size, shuffle = True)

#%% Model creation

length_mems_1 = int(1650 - min(wavelength))
length_mems_2 = int(max(wavelength) - 1750)
# vae = SpectraVAE_Double_Mems(length_mems_1, length_mems_2, hidden_space_dimension, print_var = True)
vae = SpectraVAE_Double_Mems_Conv(length_mems_1, length_mems_2, hidden_space_dimension, print_var = print_var, use_as_autoencoder = use_as_autoencoder)

optimizer = torch.optim.AdamW(vae.parameters(), lr = learning_rate, weight_decay = 1e-3)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

total_loss_train = []
total_loss_test = []

recon_loss_train = []
recon_loss_test = []

kl_loss_train = []
kl_loss_test = []

#%% Training cycle

for epoch in range(epochs):
    # Training phase
    if(use_as_autoencoder):
        tmp_loss_train_recon = advanceEpochV3(vae, device, train_dataloader, optimizer, is_train = True, double_mems = True)
        total_loss_train.append(float(tmp_loss_train_recon))
    else:
        tmp_loss_train_total, tmp_loss_train_recon, tmp_loss_train_kl = advanceEpochV2(vae, device, train_dataloader, optimizer, is_train = True, alpha = alpha, beta = beta)
        total_loss_train.append(float(tmp_loss_train_total))
        recon_loss_train.append(float(tmp_loss_train_recon))
        kl_loss_train.append(float(tmp_loss_train_kl))
    
    # Testing phase 
    if(use_as_autoencoder):
        tmp_loss_test_recon = advanceEpochV3(vae, device, train_dataloader, optimizer, is_train = False, double_mems = True)
        total_loss_test.append(float(tmp_loss_test_recon))
    else:
        tmp_loss_test_total, tmp_loss_test_recon, tmp_loss_test_kl = advanceEpochV2(vae, device, test_dataloader, optimizer, is_train = False, alpha = alpha, beta = beta)
        total_loss_test.append(float(tmp_loss_test_total))
        recon_loss_test.append(float(tmp_loss_test_recon))
        kl_loss_test.append(float(tmp_loss_test_kl))
    
    
    if(print_var and epoch % step_show == 0):
        print("Epoch: {} ({:.2f}%)".format(epoch, epoch/epochs * 100), optimizer.param_groups[0]['lr'])
      
        print("\tLoss (TRAIN)\t\t: ", float(total_loss_train[-1]))
        if not use_as_autoencoder:
            print("\t\tReconstr (TRAIN)\t: ", float(recon_loss_train[-1]))
            print("\t\tKullback (TRAIN)\t: ", float(kl_loss_train[-1]), "\n")
        
        print("\tLoss (TEST)\t\t: ", float(total_loss_test[-1]))
        if not use_as_autoencoder:
            print("\t\tReconstr (TEST)\t: ", float(tmp_loss_test_recon))
            print("\t\tKullback (TEST)\t: ", float(tmp_loss_test_kl), "\n")
        
      
        print("- - - - - - - - - - - - - - - - - - - - - - - - ")
    
    # Learning rate scheduler
    scheduler.step()
    
    n_samples = -1
    s = 1
    alpha = 0.6
    dimensionality_reduction = 'pca'

    # P.s. For the full spectra I skip the first one for reasons linked to the construction of the water gradient vector
    if use_as_autoencoder:
        visualize_latent_space_V4(full_spectra_dataset[1:], extended_water_timestamp, vae, alpha = alpha, s = s, section = 'full', n_samples = n_samples, dimensionality_reduction = dimensionality_reduction, figsize = (15, 15), device = device)
    else:
        visualize_latent_space_V3(full_spectra_dataset[1:], extended_water_timestamp, vae, resampling = False, alpha = alpha, s = s, section = 'full', n_samples = n_samples, dimensionality_reduction = dimensionality_reduction, figsize = (15, 15), device = device)
