"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)
"""

#%% Imports

import numpy as np

import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from support.training import advance_recon_loss
from support.VAE import SpectraVAE_Single_Mems, SpectraVAE_Double_Mems, sampling_latent_space

#%% Visualize loss during training

def compare_results_by_spectra(total_loss, recon_loss, kl_loss, figsize = (18, 6)):
    fig, ax = plt.subplots(1, 3, figsize = figsize)
    
    i = 0
    ax[i].plot(total_loss[i])
    ax[i].plot(recon_loss[i])
    ax[i].plot(kl_loss[i])
    ax[i].set_title("Good Spectra (TRAIN)")
    ax[i].legend(["Total Loss", "Reconstruction loss", "KL Loss"])
    ax[i].set_xlabel("Epochs")
    ax[i].set_yscale('log')
    
    i = 1
    ax[i].plot(total_loss[i])
    ax[i].plot(recon_loss[i])
    ax[i].plot(kl_loss[i])
    ax[i].set_title("Good Spectra (TEST)")
    ax[i].legend(["Total Loss", "Reconstruction loss", "KL Loss"])
    ax[i].set_xlabel("Epochs")
    ax[i].set_yscale('log')
    
    i = 2
    ax[i].plot(total_loss[i])
    ax[i].plot(recon_loss[i])
    ax[i].plot(kl_loss[i])
    ax[i].set_title("Bad Spectra ")
    ax[i].legend(["Total Loss", "Reconstruction loss", "KL Loss"])
    ax[i].set_xlabel("Epochs")
    ax[i].set_yscale('log')
    
    # plt.title("Comparison by Spectra")
    
    
def compare_results_by_loss(total_loss, recon_loss, kl_loss, figsize = (18, 6)):
    fig, ax = plt.subplots(1, 3, figsize = figsize)
    
    i = 0
    ax[i].plot(total_loss[0])
    ax[i].plot(total_loss[1])
    ax[i].plot(total_loss[2])
    ax[i].set_title("Total loss")
    ax[i].legend(["Good Spectra (TRAIN)", "Good Spectra (TEST)", "Bad Spectra"])
    ax[i].set_xlabel("Epochs")
    ax[i].set_yscale('log')
    
    i = 1
    ax[i].plot(recon_loss[0])
    ax[i].plot(recon_loss[1])
    ax[i].plot(recon_loss[2])
    ax[i].set_title("Reconstruction loss")
    ax[i].legend(["Good Spectra (TRAIN)", "Good Spectra (TEST)", "Bad Spectra"])
    ax[i].set_xlabel("Epochs")
    ax[i].set_yscale('log')
    
    i = 2
    ax[i].plot(kl_loss[0])
    ax[i].plot(kl_loss[1])
    ax[i].plot(kl_loss[2])
    ax[i].set_title("KL loss")
    ax[i].legend(["Good Spectra (TRAIN)", "Good Spectra (TEST)", "Bad Spectra"])
    ax[i].set_xlabel("Epochs")
    ax[i].set_yscale('log')
    
    # plt.title("Comparison by Loss")
    
#%% Histogram

def draw_hist_loss(good_spectra_dataset_train, good_spectra_dataset_validation, bad_spectra_dataset, vae,  device = torch.device("cpu"), batch_size = 50, n_spectra = -1, figsize = (10, 8), labels = ["Train", "Validation", "Bad"]):
    vae.to(device)
    vae.eval()
    
    good_train_dataloader = DataLoader(good_spectra_dataset_train, batch_size = batch_size, shuffle = True)
    good_spectra_train_loss = float(compute_average_loss_given_dataloader(good_train_dataloader, vae, device, n_spectra).cpu())
    
    good_validation_dataloader = DataLoader(good_spectra_dataset_validation, batch_size = batch_size, shuffle = True)
    good_spectra_validation_loss = float(compute_average_loss_given_dataloader(good_validation_dataloader, vae, device, n_spectra).cpu())
    
    bad_dataloader = DataLoader(bad_spectra_dataset, batch_size = batch_size, shuffle = True)
    bad_spectra_loss = float(compute_average_loss_given_dataloader(bad_dataloader, vae, device, n_spectra).cpu())
    
    tmp_loss = [good_spectra_train_loss, good_spectra_validation_loss, bad_spectra_loss]
    plt.figure(figsize = figsize)
    color = ['C0', 'orange', 'red']
    plt.bar(labels, tmp_loss, color = color)
    plt.title("Reconstruciton Error")
    
    
def compute_average_loss_given_dataloader(dataloader, model, device, n_spectra):
    
    total_loss = 0
    tot_elements = 0
    
    for sample_data_batch in dataloader:
        x = sample_data_batch.to(device)
        model.to(device)
        
        # Single mems
        # tmp_results = model(x)
        # x_loss = advance_recon_loss(x, tmp_results[0], tmp_results[1])
       
        # Double mems
        x1 = x[:, 0:300]
        x2 = x[:, (- 1 - 400):-1]
        x_r_1, log_var_r_1, x_r_2, log_var_r_2, mu_z, log_var_z = model(x1, x2)
        
        x_r = torch.cat((x_r_1, x_r_2), 1)
        x = torch.cat((x1,x2), 1)
        log_var_r = torch.cat((log_var_r_1, log_var_r_2), 0)
        sigma_r = torch.sqrt(torch.exp(log_var_r))
        
        x_loss = advance_recon_loss(x, x_r, sigma_r)

        
        tot_elements += x.shape[0]
        if(n_spectra > 0 and tot_elements >= n_spectra):
            tmp_n_elements = x.shape[0] - (tot_elements - n_spectra)
            total_loss += torch.sum(x_loss[0:tmp_n_elements])
            break
        else:
            total_loss += torch.sum(x_loss)
            

    
    return total_loss/tot_elements
        
#%%

def visualize_latent_space_V1(dataset_list, vae, resampling, alpha = 0.8, s = 0.3, section = 'full', n_samples = -1):
    vae.cpu()
    if(section == 'full'):  vae2 = SpectraVAE_Double_Mems(300, 400, 2, print_var = True)
    else: vae2 = SpectraVAE_Single_Mems(dataset_list[0][0].shape[0], 2, print_var = True)
        
    marker = 'x'

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    
    color_list = ['c0', 'green', 'orange', 'red']
    
    for dataset, color in zip(dataset_list, color_list):
        if(n_samples <= 0 or n_samples > len(dataset)): n_samples = len(dataset)
        
        if(section == 'full'):
            x1 = dataset[0:n_samples][:, 0:300]
            x2 = dataset[0:n_samples][:, (- 1 - 400):-1]
            x_r_1, log_var_r_1, x_r_2, log_var_r_2, mu_z, log_var_z = vae2(x1, x2)
            
            x_r = torch.cat((x_r_1, x_r_2), 1)
            x = torch.cat((x1,x2), 1)
            log_var_r = torch.cat((log_var_r_1, log_var_r_2), 0)
            sigma_r = torch.sqrt(torch.exp(log_var_r))
        else:
            x_r, log_var_r, mu_z, log_var_z = vae2(dataset[:])
        
        if(resampling): 
            # p = sampling_latent_space(mu, log_var)
            p = torch.normal(mu_z, torch.sqrt(torch.exp(log_var_z))).detach().numpy()
            ax[0].scatter(p[:, 0], p[:, 1], alpha = alpha, marker = marker, s = s)
        else: 
            ax[0].scatter(mu_z[:, 0].detach().numpy(), mu_z[:, 1].detach().numpy(), alpha = alpha, marker = marker, s = s)
        
        if(section == 'full'):
            x1 = dataset[0:n_samples][:, 0:300]
            x2 = dataset[0:n_samples][:, (- 1 - 400):-1]
            x_r_1, log_var_r_1, x_r_2, log_var_r_2, mu_z, log_var_z = vae(x1, x2)
            
            x_r = torch.cat((x_r_1, x_r_2), 1)
            x = torch.cat((x1,x2), 1)
            log_var_r = torch.cat((log_var_r_1, log_var_r_2), 0)
            sigma_r = torch.sqrt(torch.exp(log_var_r))
        else:
            x_r, log_var_r, mu_z, log_var_z = vae(dataset[:])
        if(resampling): 
            # p = sampling_latent_space(mu, log_var)
            p = torch.normal(mu_z, torch.sqrt(torch.exp(log_var_z))).detach().numpy()
            ax[1].scatter(p[:, 0], p[:, 1], alpha = alpha, marker = marker, s = s)
        else: 
            ax[1].scatter(mu_z[:, 0].detach().numpy(), mu_z[:, 1].detach().numpy(), alpha = alpha, marker = marker, s = s)
    

    ax[0].set_title("Untrained VAE")
    ax[0].legend(["Good Spectra (TRAIN)", "Good Spectra (TEST)", "Good Spectra (VAL)", "Bad Spectra"])
    
    ax[1].set_title("Trained VAE")
    ax[1].legend(["Good Spectra (TRAIN)", "Good Spectra (TEST)", "Good Spectra (VAL)", "Bad Spectra"])
