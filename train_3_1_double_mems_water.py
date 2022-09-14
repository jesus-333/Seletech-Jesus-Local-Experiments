# Imports 

import sys
sys.path.insert(1, 'support')

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from support.datasets import load_spectra_data, load_water_data, create_extended_water_vector, choose_spectra_based_on_water_V1
from support.datasets import PytorchDatasetPlantSpectra_V1
from support.VAE import SpectraVAE_Double_Mems, AttentionVAE
from support.VAE_Conv import SpectraVAE_Double_Mems_Conv
from support.training import advanceEpochV2, VAE_loss, advance_recon_loss, advanceEpochV3
from support.visualization import compare_results_by_spectra, compare_results_by_loss, draw_hist_loss
from support.visualization import visualize_latent_space_V1, visualize_latent_space_V2, visualize_latent_space_V3

#%% Parameters

normalize_trials = 1

hidden_space_dimension = 2
batch_size = 75
epochs = 100
learning_rate = 1e-3
alpha = 1 # Hyperparemeter to fine tuning the value of the reconstruction error
beta = 3 # Hyperparemeter to fine tuning the value of the KL Loss

time_interval_start = 45
time_interval_end = 360

print_var = True
step_show = 2

plot_latent_space_during_training = False

use_as_autoencoder = False

use_cnn = False

use_attention = False
embedding_size = 64

#%% Load data

if use_attention and use_cnn: raise ValueError("Both use_attention and use_cnn are True. You can't use both together because I don't have implemented attention for CNN (yet)")

# Sepctra
spectra_plants_numpy, wavelength, timestamp = load_spectra_data("data/[2021-08-05_to_11-26]All_PlantSpectra.csv", normalize_trials)

# Water
water_data, water_timestamp = load_water_data("data/[2021-08-05_to_11-26]PlantTest_Notes.csv")
extended_water_timestamp = create_extended_water_vector(water_timestamp, water_data, timestamp)

# Due to the fact that I have much more bad spectra in this way I use them to train the network.
# Cause I'm lazy I only flip the variable at the beggining and noth change all the variable along the script
# bad_idx = water, good_idx = NO water
bad_idx, good_idx = choose_spectra_based_on_water_V1(extended_water_timestamp, time_interval_start = time_interval_start, time_interval_end = time_interval_end)
labels = ["NO Water (Train)", "NO Water (Test)", "Water"]

# "Right" order
# good_idx, bad_idx = choose_spectra_based_on_water(extended_water_timestamp, time_interval_start = 60, time_interval_end = 300)
# labels = ["Water (Train)", "Water (Validation)", "NO Water"]

#%% Dataset creation
wavelength_idx = wavelength > 1

good_spectra_dataset = PytorchDatasetPlantSpectra_V1(spectra_plants_numpy[good_idx, :], used_in_cnn = use_cnn)
good_spectra_dataset_train = good_spectra_dataset[0:int(len(good_spectra_dataset) * 0.6)]
good_spectra_dataset_test = good_spectra_dataset[int(len(good_spectra_dataset) * 0.6):int(len(good_spectra_dataset) * 0.8)]
good_spectra_dataset_validation = good_spectra_dataset[int(len(good_spectra_dataset) * 0.8):]
bad_spectra_dataset = PytorchDatasetPlantSpectra_V1(spectra_plants_numpy[bad_idx, :], used_in_cnn = use_cnn)
print("N good spectra (Train)= ", len(good_spectra_dataset_train))
print("N good spectra (Test) = ", len(good_spectra_dataset_test))
print("N good spectra (Val)  = ", len(good_spectra_dataset_validation))
print("N bad spectra         = ", len(bad_spectra_dataset))
print("Total spectra         = ", len(good_spectra_dataset_train) + len(good_spectra_dataset_test) + len(good_spectra_dataset_validation) + len(bad_spectra_dataset), "\n")

good_dataloader_train = DataLoader(good_spectra_dataset_train, batch_size = batch_size, shuffle = True)
good_dataloader_test = DataLoader(good_spectra_dataset_test, batch_size = batch_size, shuffle = True)
bad_dataloader = DataLoader(bad_spectra_dataset, batch_size = batch_size, shuffle = True)

#%% Training

length_mems_1 = int(1650 - min(wavelength))
length_mems_2 = int(max(wavelength) - 1750)
if use_cnn:
    vae = SpectraVAE_Double_Mems_Conv(length_mems_1, length_mems_2, hidden_space_dimension, print_var = print_var, use_as_autoencoder = use_as_autoencoder)
else:
    if use_attention:
        vae = AttentionVAE(length_mems_1, length_mems_2, hidden_space_dimension, embedding_size,
                                     print_var = print_var, use_as_autoencoder = use_as_autoencoder )
    else:
        vae = SpectraVAE_Double_Mems(length_mems_1, length_mems_2, hidden_space_dimension, 
                                     print_var = print_var, use_as_autoencoder = use_as_autoencoder )
optimizer = torch.optim.AdamW(vae.parameters(), lr = learning_rate, weight_decay = 1e-3)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

total_loss_good_train = []
total_loss_good_test = []
total_loss_bad = []

recon_loss_good_train = []
recon_loss_good_test = []
recon_loss_bad = []

kl_loss_good_train = []
kl_loss_good_test = []
kl_loss_bad = []


for epoch in range(epochs):
    # Training phase (GOOD Spectra)
    if(use_as_autoencoder):
        tmp_loss_good_recon = advanceEpochV3(vae, device, good_dataloader_train, optimizer, is_train = True, double_mems = True)
        total_loss_good_train.append(float(tmp_loss_good_recon))
    else:
        tmp_loss_good_total, tmp_loss_good_recon, tmp_loss_good_kl = advanceEpochV2(vae, device, good_dataloader_train, optimizer, is_train = True, alpha = alpha, beta = beta)
        total_loss_good_train.append(float(tmp_loss_good_total))
        recon_loss_good_train.append(float(tmp_loss_good_recon))
        kl_loss_good_train.append(float(tmp_loss_good_kl))
    
    # Testing phase (GOOD Spectra)
    if(use_as_autoencoder):
        tmp_loss_good_recon = advanceEpochV3(vae, device, good_dataloader_test, optimizer, is_train = True, double_mems = True)
        total_loss_good_test.append(float(tmp_loss_good_recon))
    else:
        tmp_loss_good_total, tmp_loss_good_recon, tmp_loss_good_kl = advanceEpochV2(vae, device, good_dataloader_test, optimizer, is_train = False, alpha = alpha, beta = beta)
        total_loss_good_test.append(float(tmp_loss_good_total))
        recon_loss_good_test.append(float(tmp_loss_good_recon))
        kl_loss_good_test.append(float(tmp_loss_good_kl))
    
    # Testing phase (BAD Spectra)
    if(use_as_autoencoder):
        tmp_loss_bad_recon = advanceEpochV3(vae, device, bad_dataloader, optimizer, is_train = True, double_mems = True)
        total_loss_bad.append(float(tmp_loss_bad_recon))
    else:
        tmp_loss_bad_total, tmp_loss_bad_recon, tmp_loss_bad_kl = advanceEpochV2(vae, device, bad_dataloader, optimizer, is_train = False, alpha = alpha, beta = beta)
        total_loss_bad.append(float(tmp_loss_bad_total))
        recon_loss_bad.append(float(tmp_loss_bad_recon))
        kl_loss_bad.append(float(tmp_loss_bad_kl))
    
    if(print_var and epoch % step_show == 0):
        print("Epoch: {} ({:.2f}%)".format(epoch, epoch/epochs * 100))
      
        print("\tLoss (GOOD)(TRAIN)\t: ", float(total_loss_good_train[-1]))
        if not use_as_autoencoder:
            print("\t\tReconstr (GOOD)(TRAIN)\t: ", float(recon_loss_good_train[-1]))
            print("\t\tKullback (GOOD)(TRAIN)\t: ", float(kl_loss_good_train[-1]), "\n")
        
        print("\tLoss (GOOD)(TEST)\t\t: ", float(total_loss_good_test[-1]))
        if not use_as_autoencoder:
            print("\t\tReconstr (GOOD)(TEST)\t: ", float(tmp_loss_good_recon))
            print("\t\tKullback (GOOD)(TEST)\t: ", float(tmp_loss_good_kl), "\n")
        
        print("\tLoss (BAD)\t\t\t: ", float(total_loss_bad[-1]))
        if not use_as_autoencoder:
            print("\t\tReconstr (BAD)\t: ", float(tmp_loss_bad_recon))
            print("\t\tKullback (BAD)\t: ", float(tmp_loss_bad_kl), "\n")
      
        print("- - - - - - - - - - - - - - - - - - - - - - - - ")
        
    scheduler.step()
    
    if not use_as_autoencoder and plot_latent_space_during_training:
    
        n_samples = -1
        s = 1
        alpha = 0.6
        dimensionality_reduction = 'pca'
        dataset_list = [good_spectra_dataset_train, good_spectra_dataset_test, good_spectra_dataset_validation]
        visualize_latent_space_V2(dataset_list, vae, resampling = False, alpha = alpha, s = s, section = 'full', n_samples = n_samples, dimensionality_reduction = dimensionality_reduction, figsize = (15, 15), device = device)
        
#%% Plot Loss vs Epochs
figsize = (18, 6)

if not use_as_autoencoder:
    total_loss = [total_loss_good_train, total_loss_good_test, total_loss_bad]
    recon_loss = [recon_loss_good_train, recon_loss_good_test, recon_loss_bad]
    kl_loss = [kl_loss_good_train, kl_loss_good_test, kl_loss_bad]
    
    compare_results_by_spectra(total_loss, recon_loss, kl_loss, labels, figsize)
    
    compare_results_by_loss(total_loss, recon_loss, kl_loss, labels, figsize)

else:
    plt.figure(figsize = figsize)
    plt.plot(np.asarray(total_loss_good_train))
    plt.plot(np.asarray(total_loss_good_test))
    plt.plot(np.asarray(total_loss_bad))
    plt.yscale('log')
    plt.legend(labels)
    
#%% Plot Loss Average Hist

figsize = (15, 12)
n_spectra = -1

draw_hist_loss(good_spectra_dataset_train, good_spectra_dataset_validation, bad_spectra_dataset, vae,  device = device, batch_size = 50, n_spectra = n_spectra, figsize = figsize, labels = labels, use_as_autoencoder = use_as_autoencoder)
# params = {'mathtext.default': 'regular', 'font.size': 20}       
# plt.rcParams.update(params)
plt.tight_layout()

#%% Latent space visualization

if not use_as_autoencoder:
    n_samples = -1
    s = 1
    alpha = 0.6
    dimensionality_reduction = 'pca'
    
    dataset_list = [good_spectra_dataset_train, good_spectra_dataset_test, good_spectra_dataset_validation, bad_spectra_dataset]
    
    visualize_latent_space_V1(dataset_list, vae, resampling = False, alpha = alpha, s = s, section = 'full', n_samples = n_samples, hidden_space_dimension = hidden_space_dimension, dimensionality_reduction = dimensionality_reduction)
    
    # visualize_latent_space_V1(dataset_list, vae, resampling = True, alpha = alpha, s = s, section = 'full', n_samples = n_samples, hidden_space_dimension = hidden_space_dimension, dimensionality_reduction = dimensionality_reduction)
    
    # visualize_latent_space_V2(dataset_list, vae, resampling = False, alpha = alpha, s = s, section = 'full', n_samples = n_samples, dimensionality_reduction = dimensionality_reduction, figsize = (15, 15))

