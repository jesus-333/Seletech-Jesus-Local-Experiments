# Imports 

import sys
sys.path.insert(1, 'support')

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from support.datasets import load_spectra_data, load_water_data, create_extended_water_vector, choose_spectra_based_on_water_V1, choose_spectra_based_on_water_V2
from support.datasets import PytorchDatasetPlantSpectra_V1
from support.VAE import SpectraVAE_Double_Mems
from support.VAE_Conv import SpectraVAE_Double_Mems_Conv
from support.training import advanceEpochV2, VAE_loss, advance_recon_loss, advanceEpochV3
from support.visualization import compare_results_by_spectra, compare_results_by_loss, draw_hist_loss, compute_recon_loss_given_dataset_autoencoder
from support.visualization import visualize_latent_space_V1, visualize_latent_space_V2, visualize_latent_space_V3

#%% Parameters

normalize_trials = 1

hidden_space_dimension = 2
batch_size = 65
epochs = 200
learning_rate = 1e-3
alpha = 1 # Hyperparemeter to fine tuning the value of the reconstruction error
beta = 3 # Hyperparemeter to fine tuning the value of the KL Loss

time_interval_start = 45
time_interval_end = 360

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
minute_windows = 8 * 60
minute_shift = 5
avg_spectra_matrix, count_water = choose_spectra_based_on_water_V2(spectra_plants_numpy, extended_water_timestamp, minute_windows = minute_windows,  minute_shift = minute_shift)
if(print_var):
    print("Length (in minutes) of the windows: ", minute_windows)
    print("Shift (in minutes) of the windows:  ", minute_shift)
    
    print("Counts of spectra:")
    for label in set(count_water):
        print("\tN. Spectra: {}\t Water Count:{}".format(len(count_water[count_water == label]), label))

# Due to the fact that I have much more bad spectra in this way I use them to train the network.
# Cause I'm lazy I only flip the variable at the beggining and noth change all the variable along the script
# bad_idx = water, good_idx = NO water
# bad_idx, good_idx = choose_spectra_based_on_water_V1(extended_water_timestamp, time_interval_start = time_interval_start, time_interval_end = time_interval_end)
bad_idx = count_water == 0
good_idx_1 = count_water == 1
good_idx_2 = count_water == 2
labels = ["NO Water (Train)", "NO Water (Test)", "Water"]

# "Right" order
# good_idx, bad_idx = choose_spectra_based_on_water(extended_water_timestamp, time_interval_start = 60, time_interval_end = 300)
# labels = ["Water (Train)", "Water (Validation)", "NO Water"]

#%% Dataset creation
wavelength_idx = wavelength > 1

good_spectra_dataset_1 = PytorchDatasetPlantSpectra_V1(avg_spectra_matrix[good_idx_1, :], used_in_cnn = use_cnn)
good_spectra_dataset_2 = PytorchDatasetPlantSpectra_V1(avg_spectra_matrix[good_idx_2, :], used_in_cnn = use_cnn)
bad_spectra_dataset = PytorchDatasetPlantSpectra_V1(avg_spectra_matrix[bad_idx, :], used_in_cnn = use_cnn)
bad_spectra_dataset_train = bad_spectra_dataset[0:int(len(bad_spectra_dataset) * 0.8)]
bad_spectra_dataset_test = bad_spectra_dataset[int(len(bad_spectra_dataset) * 0.8):]
print("N good spectra (1) = ", len(good_spectra_dataset_1))
print("N good spectra (2) = ", len(good_spectra_dataset_2))
print("N bad spectra      = ", len(bad_spectra_dataset))
print("Total spectra      = ", len(good_spectra_dataset_1) + len(good_spectra_dataset_2) + len(bad_spectra_dataset), "\n")

bad_dataloader_train = DataLoader(bad_spectra_dataset_train, batch_size = batch_size, shuffle = True)
bad_dataloader_test = DataLoader(bad_spectra_dataset_test, batch_size = batch_size, shuffle = True)
good_dataloader_1 = DataLoader(good_spectra_dataset_1, batch_size = batch_size, shuffle = True)
good_dataloader_2 = DataLoader(good_spectra_dataset_2, batch_size = batch_size, shuffle = True)

#%% Training

length_mems_1 = int(1650 - min(wavelength))
length_mems_2 = int(max(wavelength) - 1750)
if use_cnn:
    vae = SpectraVAE_Double_Mems_Conv(length_mems_1, length_mems_2, hidden_space_dimension, print_var = print_var, use_as_autoencoder = use_as_autoencoder)
else:
    vae = SpectraVAE_Double_Mems(length_mems_1, length_mems_2, hidden_space_dimension, print_var = print_var, use_as_autoencoder = use_as_autoencoder )

    
optimizer = torch.optim.AdamW(vae.parameters(), lr = learning_rate, weight_decay = 1e-3)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

total_loss_bad_train = []
total_loss_bad_test = []
total_loss_good_1 = []
total_loss_good_2 = []

recon_loss_bad_train = []
recon_loss_bad_test = []
recon_loss_good_1 = []
recon_loss_good_2 = []

kl_loss_bad_train = []
kl_loss_bad_test = []
kl_loss_good_1 = []
kl_loss_good_2 = []


for epoch in range(epochs):
    # Training phase (BAS Spectra)
    if(use_as_autoencoder):
        tmp_loss_bad_recon = advanceEpochV3(vae, device, bad_dataloader_train, optimizer, is_train = True, double_mems = True)
        total_loss_bad_train.append(float(tmp_loss_bad_recon))
    else:
        tmp_loss_bad_total, tmp_loss_bad_recon, tmp_loss_bad_kl = advanceEpochV2(vae, device, bad_dataloader_train, optimizer, is_train = True, alpha = alpha, beta = beta)
        total_loss_bad_train.append(float(tmp_loss_bad_total))
        recon_loss_bad_train.append(float(tmp_loss_bad_recon))
        kl_loss_bad_train.append(float(tmp_loss_bad_kl))
    
    # Testing phase (BAD Spectra)
    if(use_as_autoencoder):
        tmp_loss_bad_recon = advanceEpochV3(vae, device, bad_dataloader_test, optimizer, is_train = True, double_mems = True)
        total_loss_bad_test.append(float(tmp_loss_bad_recon))
    else:
        tmp_loss_bad_total, tmp_loss_bad_recon, tmp_loss_bad_kl = advanceEpochV2(vae, device, bad_dataloader_test, optimizer, is_train = False, alpha = alpha, beta = beta)
        total_loss_bad_test.append(float(tmp_loss_bad_total))
        recon_loss_bad_test.append(float(tmp_loss_bad_recon))
        kl_loss_bad_test.append(float(tmp_loss_bad_kl))
    
    # Testing phase (GOOD Spectra 1)
    if(use_as_autoencoder):
        tmp_loss_good_recon_1 = advanceEpochV3(vae, device, good_dataloader_1, optimizer, is_train = True, double_mems = True)
        total_loss_good_1.append(float(tmp_loss_good_recon_1))
    else:
        tmp_loss_good_total_1, tmp_loss_good_recon_1, tmp_loss_good_kl_1 = advanceEpochV2(vae, device, good_dataloader_1, optimizer, is_train = False, alpha = alpha, beta = beta)
        total_loss_good_1.append(float(tmp_loss_good_total_1))
        recon_loss_good_1.append(float(tmp_loss_good_recon_1))
        kl_loss_good_1.append(float(tmp_loss_good_kl_1))
    
    # Testing phase (GOOD Spectra 2)
    if(use_as_autoencoder):
        tmp_loss_good_recon_2 = advanceEpochV3(vae, device, good_dataloader_2, optimizer, is_train = True, double_mems = True)
        total_loss_good_2.append(float(tmp_loss_good_recon_2))
    else:
        tmp_loss_good_total_2, tmp_loss_good_recon_2, tmp_loss_good_kl_2 = advanceEpochV2(vae, device, good_dataloader_1, optimizer, is_train = False, alpha = alpha, beta = beta)
        total_loss_good_2.append(float(tmp_loss_good_total_2))
        recon_loss_good_2.append(float(tmp_loss_good_recon_2))
        kl_loss_good_2.append(float(tmp_loss_good_kl_2))
        
    
    if(print_var and epoch % step_show == 0):
        print("Epoch: {} ({:.2f}%)".format(epoch, epoch/epochs * 100), optimizer.param_groups[0]['lr'])
      
        print("\tLoss (BAD)(TRAIN)\t\t: ", float(total_loss_bad_train[-1]))
        if not use_as_autoencoder:
            print("\t\tReconstr (BAD)(TRAIN)\t: ", float(recon_loss_bad_train[-1]))
            print("\t\tKullback (BAD)(TRAIN)\t: ", float(kl_loss_bad_train[-1]), "\n")
        
        print("\tLoss (BAD)(TEST)\t\t: ", float(total_loss_bad_test[-1]))
        if not use_as_autoencoder:
            print("\t\tReconstr (BAD)(TEST)\t: ", float(tmp_loss_bad_recon))
            print("\t\tKullback (BAD)(TEST)\t: ", float(tmp_loss_bad_kl), "\n")
        
        print("\tLoss (GOOD 1)\t\t: ", float(total_loss_good_1[-1]))
        if not use_as_autoencoder:
            print("\t\tReconstr (GOOD 1)\t: ", float(tmp_loss_good_recon_1))
            print("\t\tKullback (GOOD 1)\t: ", float(tmp_loss_good_kl_1), "\n")
            
        print("\tLoss (GOOD 2)\t\t: ", float(total_loss_good_2[-1]))
        if not use_as_autoencoder:
            print("\t\tReconstr (GOOD 1)\t: ", float(tmp_loss_good_recon_2))
            print("\t\tKullback (GOOD 1)\t: ", float(tmp_loss_good_kl_2), "\n")
        
        print("- - - - - - - - - - - - - - - - - - - - - - - - ")
        
    scheduler.step()
    
    # if not use_as_autoencoder:
    
    #     n_samples = -1
    #     s = 1
    #     alpha = 0.6
    #     dimensionality_reduction = 'pca'
    #     dataset_list = [good_spectra_dataset_train, good_spectra_dataset_test, good_spectra_dataset_validation]
    #     visualize_latent_space_V2(dataset_list, vae, resampling = False, alpha = alpha, s = s, section = 'full', n_samples = n_samples, dimensionality_reduction = dimensionality_reduction, figsize = (15, 15), device = device)
        
#%%
figsize = (6, 12)

# def compute_loss_given_dataset_autoencoder(dataset, model, device, compute_std = False):
#     length_mems_1 = 300
#     length_mems_2 = 400
#     loss_function = torch.nn.MSELoss()
    
#     model = model.to(device)
    
#     x = dataset[:].to(device)
    
#     x1 = x[:, 0:length_mems_1]
#     x2 = x[:, (- 1 - length_mems_2):-1]
    
#     x1_r, x2_r, z = vae(x1, x2)
    
#     x = torch.cat((x1,x2), -1)
#     x_r = torch.cat((x1_r,x2_r), -1)
    
#     loss = loss_function(x, x_r)
    
#     return loss

if(use_as_autoencoder): 
    train_loss, train_std = compute_recon_loss_given_dataset_autoencoder(bad_spectra_dataset_train, vae, device, compute_std = True)
    test_loss, test_std = compute_recon_loss_given_dataset_autoencoder(bad_spectra_dataset_test, vae, device, compute_std = True)
    good_loss_1, good_loss_1_std = compute_recon_loss_given_dataset_autoencoder(good_spectra_dataset_1, vae, device, compute_std = True)
    good_loss_2, good_loss_2_std = compute_recon_loss_given_dataset_autoencoder(good_spectra_dataset_2, vae, device, compute_std = True)
    
    tmp_loss = [train_loss.cpu().detach(), test_loss.cpu().detach(), good_loss_1.cpu().detach(), good_loss_2.cpu().detach() ]
    tmp_loss = [test_loss.cpu().detach(), good_loss_1.cpu().detach(), good_loss_2.cpu().detach() ]
    
    yerr = [train_std.cpu().detach(), test_std.cpu().detach(), good_loss_1_std.cpu().detach(), good_loss_2_std.cpu().detach()]
    yerr = [test_std.cpu().detach(), good_loss_1_std.cpu().detach(), good_loss_2_std.cpu().detach()]
    
    plt.figure(figsize = figsize)
    color = ['C0', 'orange', 'red', 'blue']
    labels = ['BAD TRAIN', 'BAD TEST', 'GOOD 1', 'GOOD 2']
    labels = ['BAD TEST', 'GOOD 1', 'GOOD 2']
    plt.bar(labels, tmp_loss, yerr = yerr, color = color)
    plt.title("Reconstruciton Error")
    plt.show()
