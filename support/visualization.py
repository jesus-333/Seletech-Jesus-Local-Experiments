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

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

#%% Visualize loss during training

def compare_results_by_spectra(total_loss, recon_loss, kl_loss, labels, figsize = (18, 6)):
    fig, ax = plt.subplots(1, 3, figsize = figsize)
    
    i = 0
    ax[i].plot(total_loss[i])
    ax[i].plot(recon_loss[i])
    ax[i].plot(kl_loss[i])
    ax[i].set_title(labels[i])
    ax[i].legend(["Total Loss", "Reconstruction loss", "KL Loss"])
    ax[i].set_xlabel("Epochs")
    ax[i].set_yscale('log')
    
    i = 1
    ax[i].plot(total_loss[i])
    ax[i].plot(recon_loss[i])
    ax[i].plot(kl_loss[i])
    ax[i].set_title(labels[i])
    ax[i].legend(["Total Loss", "Reconstruction loss", "KL Loss"])
    ax[i].set_xlabel("Epochs")
    ax[i].set_yscale('log')
    
    i = 2
    ax[i].plot(total_loss[i])
    ax[i].plot(recon_loss[i])
    ax[i].plot(kl_loss[i])
    ax[i].set_title(labels[i])
    ax[i].legend(["Total Loss", "Reconstruction loss", "KL Loss"])
    ax[i].set_xlabel("Epochs")
    ax[i].set_yscale('log')
    
    # plt.title("Comparison by Spectra")
    plt.show()
    
    
def compare_results_by_loss(total_loss, recon_loss, kl_loss, labels, figsize = (18, 6)):
    fig, ax = plt.subplots(1, 3, figsize = figsize)
    
    i = 0
    ax[i].plot(total_loss[0])
    ax[i].plot(total_loss[1])
    ax[i].plot(total_loss[2])
    ax[i].set_title("Total loss")
    ax[i].legend(labels)
    ax[i].set_xlabel("Epochs")
    ax[i].set_yscale('log')
    
    i = 1
    ax[i].plot(recon_loss[0])
    ax[i].plot(recon_loss[1])
    ax[i].plot(recon_loss[2])
    ax[i].set_title("Reconstruction loss")
    ax[i].legend(labels)
    ax[i].set_xlabel("Epochs")
    ax[i].set_yscale('log')
    
    i = 2
    ax[i].plot(kl_loss[0])
    ax[i].plot(kl_loss[1])
    ax[i].plot(kl_loss[2])
    ax[i].set_title("KL loss")
    ax[i].legend(labels)
    ax[i].set_xlabel("Epochs")
    ax[i].set_yscale('log')
    
    # plt.title("Comparison by Loss")
    plt.show()
    
#%% Histogram

def draw_hist_loss(good_spectra_dataset_train, good_spectra_dataset_validation, bad_spectra_dataset, vae,  device = torch.device("cpu"), batch_size = 50, n_spectra = -1, figsize = (10, 8), labels = ["Train", "Validation", "Bad"], use_as_autoencoder = False):
    vae.to(device)
    vae.eval()
    
    good_train_dataloader = DataLoader(good_spectra_dataset_train, batch_size = batch_size, shuffle = True)
    good_spectra_train_loss = float(compute_average_loss_given_dataloader(good_train_dataloader, vae, device, n_spectra, use_as_autoencoder).cpu())
    
    good_validation_dataloader = DataLoader(good_spectra_dataset_validation, batch_size = batch_size, shuffle = True)
    good_spectra_validation_loss = float(compute_average_loss_given_dataloader(good_validation_dataloader, vae, device, n_spectra, use_as_autoencoder).cpu())
    
    bad_dataloader = DataLoader(bad_spectra_dataset, batch_size = batch_size, shuffle = True)
    bad_spectra_loss = float(compute_average_loss_given_dataloader(bad_dataloader, vae, device, n_spectra, use_as_autoencoder).cpu())
    
    tmp_loss = [good_spectra_train_loss, good_spectra_validation_loss, bad_spectra_loss]
    
    plt.figure(figsize = figsize)
    color = ['C0', 'orange', 'red']
    plt.bar(labels, tmp_loss, color = color)
    plt.title("Reconstruciton Error")
    plt.show()
    
    
def compute_average_loss_given_dataloader(dataloader, model, device, n_spectra, use_as_autoencoder):
    model.eval()
    
    total_loss = 0
    tot_elements = 0
    
    if(use_as_autoencoder): loss_function = torch.nn.MSELoss()
    
    for sample_data_batch in dataloader:
        x = sample_data_batch.to(device)
        model.to(device)
        
        # Single mems
        # tmp_results = model(x)
        # x_loss = advance_recon_loss(x, tmp_results[0], tmp_results[1])
       
        # Double mems
        x1 = x[:, ..., 0:300]
        x2 = x[:, ..., (- 1 - 400):-1]
        if(use_as_autoencoder):
            x_r_1, x_r_2, z = model(x1, x2)
            
            x_r = torch.cat((x_r_1, x_r_2), -1)
            x = torch.cat((x1,x2), -1)
            
            x_loss = loss_function(x, x_r)
        else:
            x_r_1, log_var_r_1, x_r_2, log_var_r_2, mu_z, log_var_z = model(x1, x2)
            
            x_r = torch.cat((x_r_1, x_r_2), -1)
            x = torch.cat((x1,x2), -1)
            log_var_r = torch.cat((log_var_r_1, log_var_r_2), -1)
            sigma_r = torch.sqrt(torch.exp(log_var_r))
            
            x_loss = advance_recon_loss(x, x_r, sigma_r)

        tot_elements += x.shape[0]
        if(n_spectra > 0 and tot_elements >= n_spectra):
            tmp_n_elements = x.shape[0] - (tot_elements - n_spectra)
            total_loss += torch.sum(x_loss[0:tmp_n_elements]) * tmp_n_elements
            break
        else:
            total_loss += torch.sum(x_loss) * sample_data_batch.shape[0]
            

    return total_loss/tot_elements


def compute_recon_loss_given_dataset_autoencoder(dataset, model, device, compute_std = False):
    length_mems_1 = 300
    length_mems_2 = 400
    loss_function = torch.nn.MSELoss()
    model = model.to(device)
    
    x = dataset[:].to(device)
    
    x1 = x[:, 0:length_mems_1]
    x2 = x[:, (- 1 - length_mems_2):-1]
    
    x1_r, x2_r, z = model(x1, x2)
    
    x = torch.cat((x1,x2), -1)
    x_r = torch.cat((x1_r,x2_r), -1)
    
    loss = torch.pow(torch.abs(x - x_r), 2)
    
    if compute_std:
        return loss.mean(), loss.std()
    else:
        return loss.mean()
    
#%%

def visualize_latent_space_V1(dataset_list, vae, resampling, alpha = 0.8, s = 0.3, section = 'full', n_samples = -1, hidden_space_dimension = 2, dimensionality_reduction = 'pca'):
    """
    Represent the latent space of the VAE and make a comparison with an untrained VAE.
    """
    
    vae.cpu()
    if(section == 'full'):  vae2 = SpectraVAE_Double_Mems(300, 400, hidden_space_dimension, print_var = True)
    else: vae2 = SpectraVAE_Single_Mems(dataset_list[0][0].shape[0], hidden_space_dimension, print_var = True)
        
    marker = 'x'

    fig, ax = plt.subplots(1, 2, figsize = (20, 10))
    
    color_list = ['c0', 'green', 'orange', 'red']
    
    for dataset, color in zip(dataset_list, color_list):     
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Untrained VAE 
        p = compute_latent_space_representation(dataset, vae2, resampling, section, n_samples, dimensionality_reduction)
        ax[0].scatter(p[:, 0], p[:, 1], alpha = alpha, marker = marker, s = s)
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Trained VAE 
        p = compute_latent_space_representation(dataset, vae, resampling, section, n_samples, dimensionality_reduction)
        ax[1].scatter(p[:, 0], p[:, 1], alpha = alpha, marker = marker, s = s)
    

    ax[0].set_title("Untrained VAE")
    ax[0].legend(["Good Spectra (TRAIN)", "Good Spectra (TEST)", "Good Spectra (VAL)", "Bad Spectra"])
    
    ax[1].set_title("Trained VAE")
    ax[1].legend(["Good Spectra (TRAIN)", "Good Spectra (TEST)", "Good Spectra (VAL)", "Bad Spectra"])
    plt.show()
    
    
def visualize_latent_space_V2(dataset_list, vae, resampling, alpha = 0.8, s = 0.3, section = 'full', n_samples = -1, dimensionality_reduction = 'pca', figsize = (13, 13), device = 'cpu'):
    """
    Represent the latent space of the VAE.
    """
    
    marker = 'x'

    fig, ax = plt.subplots(1, 1, figsize = figsize)
    
    # color_list = ['c0', 'green', 'orange', 'red']
    # color_list = ['orange', 'orange', 'orange', 'red']
    color_list = ['blue', 'green', 'orange']
    
    for dataset, color in zip(dataset_list, color_list):
        if(n_samples <= 0 or n_samples > len(dataset)): n_samples = len(dataset)
        p = compute_latent_space_representation(dataset, vae, resampling, section, n_samples, dimensionality_reduction, device)
        ax.scatter(p[:, 0], p[:, 1], alpha = alpha, marker = marker, s = s, c = color)
    
    ax.set_title("Latent Space ({})".format(dimensionality_reduction))
    # ax.legend(["Good Spectra (TRAIN)", "Good Spectra (TEST)", "Good Spectra (VAL)", "Bad Spectra"])
    
    lim = 0.75
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    plt.show()
    
    
def  visualize_latent_space_V3(full_spectra_dataset, extended_water_timestamp, vae, resampling, alpha = 0.8, s = 0.3, section = 'full', n_samples = -1, dimensionality_reduction = 'pca', figsize = (13, 13), device = 'cpu'):
    """
    Given all the spectra and the relative water timestamp vector create a scatter plor where each point is colored based on the time passed from when receive water.
    Used for the VAE
    N.b. The extended_water_timestamp is obtained from the function create_extended_water_vector in dataset.py file
    """
    
    water_gradient = compute_water_gradient_vector(extended_water_timestamp, n_samples)
    
    fig, ax = plt.subplots(1, 1, figsize = figsize)
    marker = 'x'
    p = compute_latent_space_representation(full_spectra_dataset, vae, resampling, section, n_samples, dimensionality_reduction, device)
    ax.scatter(p[:, 0], p[:, 1], alpha = alpha, marker = marker, s = s, c = water_gradient, cmap = 'Greens_r')
    
    # lim = 0.01
    # ax.set_xlim([-lim, lim])
    # ax.set_ylim([-lim, lim])
    plt.show()
    

def  visualize_latent_space_V4(full_spectra_dataset, extended_water_timestamp, autoencoder, alpha = 0.8, s = 0.3, section = 'full', n_samples = -1, dimensionality_reduction = 'pca', figsize = (13, 13), device = 'cpu'):
    """
    Given all the spectra and the relative water timestamp vector create a scatter plor where each point is colored based on the time passed from when receive water.
    Used for the autoencoder
    N.b. The extended_water_timestamp is obtained from the function create_extended_water_vector in dataset.py file
    """
    
    water_gradient = compute_water_gradient_vector(extended_water_timestamp, n_samples)
    
    fig, ax = plt.subplots(1, 1, figsize = figsize)
    marker = 'x'
    
    x1 = full_spectra_dataset[0:n_samples, ..., 0:300].to(device)
    x2 = full_spectra_dataset[0:n_samples, ..., (- 1 - 400):-1].to(device)
    _, _, p = autoencoder(x1, x2)
    p = p.cpu().detach().numpy()
    
    # If the hidden space has a dimensions higher than 2 use PCA/TSNE to reduce it to two
    if(p.shape[1] > 2): 
        if(dimensionality_reduction == 'tsne'): p = TSNE(n_components = 2, learning_rate='auto', init='random').fit_transform(p)
        if(dimensionality_reduction == 'pca'): p = PCA(n_components=2).fit_transform(p)
        
    ax.scatter(p[:, 0], p[:, 1], alpha = alpha, marker = marker, s = s, c = water_gradient, cmap = 'Greens_r')
    
    # lim = 0.01
    # ax.set_xlim([-lim, lim])
    # ax.set_ylim([-lim, lim])
    plt.show()


def compute_latent_space_representation(dataset, vae, resampling, section = 'full', n_samples = -1, dimensionality_reduction = 'pca', device = 'cpu'):
    # Check the number of samples
    if(n_samples <= 0 or n_samples > len(dataset)): n_samples = len(dataset)
    
    vae.to(device)
    
    # Compute latent space representation
    if(section == 'full'): # Double mems
        # x1 = dataset[0:n_samples][:, 0:300].to(device)
        # x2 = dataset[0:n_samples][:, (- 1 - 400):-1].to(device)
        x1 = dataset[0:n_samples, ..., 0:300].to(device)
        x2 = dataset[0:n_samples, ..., (- 1 - 400):-1].to(device)
        x_r_1, log_var_r_1, x_r_2, log_var_r_2, mu_z, log_var_z = vae(x1, x2)
        
        x_r = torch.cat((x_r_1, x_r_2), -1)
        x = torch.cat((x1,x2), -1)
        log_var_r = torch.cat((log_var_r_1, log_var_r_2), -1)
        sigma_r = torch.sqrt(torch.exp(log_var_r))
    else: # Single mems
        x_r, log_var_r, mu_z, log_var_z = vae(dataset[:])
        
    if(resampling): 
        p = torch.normal(mu_z, torch.sqrt(torch.exp(log_var_z))).cpu().detach().numpy()
    else: 
        p = mu_z.cpu().detach().numpy()
        
    # If the hidden space has a dimensions higher than 2 use PCA/TSNE to reduce it to two
    if(p.shape[1] > 2): 
        if(dimensionality_reduction == 'tsne'): p = TSNE(n_components = 2, learning_rate='auto', init='random').fit_transform(p)
        if(dimensionality_reduction == 'pca'): p = PCA(n_components=2).fit_transform(p)
            
    return p


def compute_water_gradient_vector(extended_water_timestamp, n_samples):
    if(n_samples <= 0): n_samples = len(extended_water_timestamp) - 1
    
    # Create water gradient vector
    water_gradient = np.zeros(n_samples)
    for i in range(len(water_gradient)):
        if(i == 0): pass
        if(extended_water_timestamp[i] == 1): water_gradient[i] = 0
        elif(extended_water_timestamp[i] == 0): water_gradient[i] = water_gradient[i - 1] + 1
    
    # Rescale between 0 and 1
    water_gradient /= np.max(water_gradient)
    
    return water_gradient
        
