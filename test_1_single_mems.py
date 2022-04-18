# Imports 

import sys
sys.path.insert(1, 'support')

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from support.datasets import load_spectra_data, PytorchDatasetPlantSpectra_V1
from support.VAE import SpectraVAE_Single_Mems
from support.training import advanceEpochV1, VAE_loss, advance_recon_loss
from support.visualization import compare_results_by_spectra, compare_results_by_loss, draw_hist_loss
from support.visualization import visualize_latent_space_V1

#%% Parameters

section = 'lower'
# min_threeshold = 0.64 # For lower
# min_threeshold = 0.5 # For upper
normalize_trials = 1

hidden_space_dimension = 2
batch_size = 50
epochs = 50
learning_rate = 1e-2
alpha = 1 # Hyperparemeter to fine tuning the value of the reconstruction error
beta = 1.5 # Hyperparemeter to fine tuning the value of the KL Loss

print_var = True
step_show = 2


#%%
spectra_plants_numpy, wavelength, timestamp = load_spectra_data("data/[2021-08-05_to_11-26]All_PlantSpectra.csv", normalize_trials)

if(section == 'lower'): 
    wave_idx = wavelength < 1650
    min_threeshold = 0.64
if(section == 'upper'): 
    wave_idx = wavelength > 1750
    min_threeshold = 0.5

good_idx = np.min(spectra_plants_numpy[:, wave_idx], 1) <= min_threeshold
bad_idx = np.min(spectra_plants_numpy[:, wave_idx], 1) > min_threeshold


good_spectra_dataset = PytorchDatasetPlantSpectra_V1(spectra_plants_numpy[good_idx, :])
good_spectra_dataset_train = good_spectra_dataset[0:int(len(good_spectra_dataset) * 0.6)]
good_spectra_dataset_test = good_spectra_dataset[int(len(good_spectra_dataset) * 0.6):int(len(good_spectra_dataset) * 0.8)]
good_spectra_dataset_validation = good_spectra_dataset[int(len(good_spectra_dataset) * 0.8):]
bad_spectra_dataset = PytorchDatasetPlantSpectra_V1(spectra_plants_numpy[bad_idx, :])
print("N good spectra (Train)= ", len(good_spectra_dataset_train))
print("N good spectra (Test) = ", len(good_spectra_dataset_test))
print("N good spectra (Val)  = ", len(good_spectra_dataset_validation))
print("N bad spectra         = ", len(bad_spectra_dataset))
print("Total spectra         = ", len(good_spectra_dataset_train) + len(good_spectra_dataset_test) + len(good_spectra_dataset_validation) + len(bad_spectra_dataset), "\n")

good_dataloader_train = DataLoader(good_spectra_dataset_train, batch_size = batch_size, shuffle = True)
good_dataloader_test = DataLoader(good_spectra_dataset_test, batch_size = batch_size, shuffle = True)
bad_dataloader = DataLoader(bad_spectra_dataset, batch_size = batch_size, shuffle = True)

vae = SpectraVAE_Single_Mems(good_spectra_dataset_train[0].shape[0], hidden_space_dimension, print_var = True)

optimizer = torch.optim.AdamW(vae.parameters(), lr = learning_rate, weight_decay = 1e-5)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#%%

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
    # Training phase
    tmp_loss_good_total, tmp_loss_good_recon, tmp_loss_good_kl = advanceEpochV1(vae, device, good_dataloader_train, optimizer, section, is_train = True, alpha = alpha, beta = beta)
    total_loss_good_train.append(float(tmp_loss_good_total))
    recon_loss_good_train.append(float(tmp_loss_good_recon))
    kl_loss_good_train.append(float(tmp_loss_good_kl))
    
    # Testing phase (GOOD Spectra)
    tmp_loss_good_total, tmp_loss_good_recon, tmp_loss_good_kl = advanceEpochV1(vae, device, good_dataloader_test, optimizer, section, is_train = False, alpha = alpha, beta = beta)
    total_loss_good_test.append(float(tmp_loss_good_total))
    recon_loss_good_test.append(float(tmp_loss_good_recon))
    kl_loss_good_test.append(float(tmp_loss_good_kl))
    
    # Testing phase (BAD Spectra)
    tmp_loss_bad_total, tmp_loss_bad_recon, tmp_loss_bad_kl = advanceEpochV1(vae, device, bad_dataloader, optimizer, section, is_train = False, alpha = alpha, beta = beta)
    total_loss_bad.append(float(tmp_loss_bad_total))
    recon_loss_bad.append(float(tmp_loss_bad_recon))
    kl_loss_bad.append(float(tmp_loss_bad_kl))
    
    if(print_var and epoch % step_show == 0):
        print("Epoch: {} ({:.2f}%)".format(epoch, epoch/epochs * 100))
      
        print("\tLoss (GOOD)(TRAIN)\t\t: ", float(total_loss_good_train[-1]))
        print("\t\tReconstr (GOOD)(TRAIN)\t: ", float(recon_loss_good_train[-1]))
        print("\t\tKullback (GOOD)(TRAIN)\t: ", float(kl_loss_good_train[-1]), "\n")
        
        print("\tLoss (GOOD)(TEST)\t\t: ", float(tmp_loss_good_total))
        print("\t\tReconstr (GOOD)(TEST)\t: ", float(tmp_loss_good_recon))
        print("\t\tKullback (GOOD)(TEST)\t: ", float(tmp_loss_good_kl), "\n")
        
        print("\tLoss (BAD)\t\t: ", float(tmp_loss_bad_total))
        print("\t\tReconstr (BAD)\t: ", float(tmp_loss_bad_recon))
        print("\t\tKullback (BAD)\t: ", float(tmp_loss_bad_kl), "\n")
      
        print("- - - - - - - - - - - - - - - - - - - - - - - - ")
        
        
#%%

figsize = (18, 6)

total_loss = [total_loss_good_train, total_loss_good_test, total_loss_bad]
recon_loss = [recon_loss_good_train, recon_loss_good_test, recon_loss_bad]
kl_loss = [kl_loss_good_train, kl_loss_good_test, kl_loss_bad]

compare_results_by_spectra(total_loss, recon_loss, kl_loss, figsize)

compare_results_by_loss(total_loss, recon_loss, kl_loss, figsize)

#%%
figsize = (10, 8)
n_spectra = -1

draw_hist_loss(good_spectra_dataset_train, good_spectra_dataset_validation, bad_spectra_dataset, vae,  device = device, batch_size = 50, n_spectra = n_spectra, figsize = figsize)

#%%

dataset_list = [good_spectra_dataset_train, good_spectra_dataset_test, good_spectra_dataset_validation, bad_spectra_dataset]

visualize_latent_space_V1(dataset_list, vae, resampling = False, alpha = 0.8, s = 0.3)

visualize_latent_space_V1(dataset_list, vae, resampling = True, alpha = 0.8, s = 0.3)

# vae.cpu()
# vae2 = SpectraVAE_Single_Mems(good_spectra_dataset[0].shape[0], hidden_space_dimension, print_var = True)
# alpha = 0.8
# s = 0.3
# marker = 'x'

# fig, ax = plt.subplots(1, 2, figsize=(20, 10))


# x_r, log_var_r, mu_bad, log_var_bad = vae2(bad_spectra_dataset[:])
# x_r, log_var_r, mu_good_train, log_var_good_train = vae2(good_spectra_dataset_train[:])
# x_r, log_var_r, mu_good_test, log_var_good_test = vae2(good_spectra_dataset_test[:])
# ax[0].scatter(mu_good_train[:, 0].detach().numpy(), mu_good_train[:, 1].detach().numpy(), color = 'C0', alpha = alpha, marker = marker, s = s)
# ax[0].scatter(mu_good_test[:, 0].detach().numpy(), mu_good_test[:, 1].detach().numpy(), color = 'orange', alpha = alpha, marker = marker, s = s)
# ax[0].scatter(mu_bad[:, 0].detach().numpy(), mu_bad[:, 1].detach().numpy(), color = 'red', alpha = alpha, marker = marker, s = s)
# ax[0].set_title("Untrained VAE")
# ax[0].legend(["Good Spectra (TRAIN)", "Good Spectra (TEST)", "Bad Spectra"])

# x_r, log_var_r, mu_bad, log_var_bad = vae(bad_spectra_dataset[:])
# x_r, log_var_r, mu_good_train, log_var_good_train = vae(good_spectra_dataset_train[:])
# x_r, log_var_r, mu_good_test, log_var_good_test = vae(good_spectra_dataset_test[:])
# ax[1].scatter(mu_good_train[:, 0].detach().numpy(), mu_good_train[:, 1].detach().numpy(), color = 'C0', alpha = alpha, marker = marker, s = s)
# ax[1].scatter(mu_good_test[:, 0].detach().numpy(), mu_good_test[:, 1].detach().numpy(), color = 'orange', alpha = alpha, marker = marker, s = s)
# ax[1].scatter(mu_bad[:, 0].detach().numpy(), mu_bad[:, 1].detach().numpy(), color = 'red', alpha = alpha, marker = marker, s = s)
# ax[1].set_title("Trained VAE")
# ax[1].legend(["Good Spectra (TRAIN)", "Good Spectra (TEST)", "Bad Spectra"])