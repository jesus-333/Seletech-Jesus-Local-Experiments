#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 14:43:02 2022

@author: Alberto Zancanaro (jesus)
@organization: University of Padua
"""

#%% Imports

import wandb
import torch

from support.wandb_init_V1 import make_dataloader
from support.wandb_init_V2 import load_model_from_artifact_inside_run, load_dataset_from_artifact_inside_run, add_model_to_artifact, split_dataset
from support.preprocess import choose_spectra_based_on_water_V1
from support.datasets import PytorchDatasetPlantSpectra_V1

#%% Principal function

def train_and_log_VAE_model(project_name, config):
    with wandb.init(project = project_name, job_type = "train", config = config) as run:
        config = wandb.config
          
        # Load model from artifacts
        model, model_config = load_model_from_artifact_inside_run(run, config['model_artifact_name'],
                                                    version = config['version'], model_name = 'untrained.pth')
        
        # Check if it is used as autoencoder
        config['use_as_autoencoder'] = model_config['use_as_autoencoder']
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr = config['lr'], 
                                      weight_decay = config['optimizer_weight_decay'])
        
        # Setup lr scheduler
        if config['use_scheduler'] == True:
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = config['gamma'])
        else:
            lr_scheduler = None
        
        # Print the training device
        if config['print_var']: print("Model trained on: {}".format(config['device']))
        
        # Setup the dataloader
        loader_list = load_loader(config)
        
        # Train model
        model.to(config['device'])
        wandb.watch(model, log = "all", log_freq = config['log_freq'])
        train_anomaly_model(model, optimizer, loader_list, config, lr_scheduler)

        # Save model after training
        model_artifact_name = config['model_artifact_name'] + '_trained'
        metadata = dict(training_config = config, model_config = model_config)
        model_artifact = wandb.Artifact(model_artifact_name, type = "model",
                                        description = "Trained {}:{} model".format(config['model_artifact_name'], config['version']),
                                        metadata = metadata)
        add_model_to_artifact(model, model_artifact)
        run.log_artifact(model_artifact)
        
        return model

#%% Load data for training

def load_loader(config, run):
    # Load data from dataset artifact
    data = load_dataset_from_artifact_inside_run(config['dataset_config'], run)
    
    spectra = data[0]
    extended_water_timestamp = data[3]
    
    # Divide the spectra in good (Water) and Bad (NON water)
    good_idx, bad_idx = choose_spectra_based_on_water_V1(extended_water_timestamp, time_interval_start = config['time_interval_start'], time_interval_end = config['time_interval_end'])
    
    good_spectra_dataset = PytorchDatasetPlantSpectra_V1(spectra[good_idx, :], used_in_cnn = config['use_cnn'])
    bad_spectra_dataset = PytorchDatasetPlantSpectra_V1(spectra[bad_idx, :], used_in_cnn = config['use_cnn'])
    
    # Save the idx used to divide the dataset between normal(dry/bad) and anomaly (good/water)
    # N.b. The dry spectra are more numerous
    config['dataset_config']['good_idx'] = good_idx
    config['dataset_config']['bad_idx'] = bad_idx
    
    # Divided bad dataset
    bad_dataset_train, bad_dataset_test, bad_dataset_validation, split_idx = split_dataset(bad_spectra_dataset, config['dataset_config'])
    config['dataset_config']['bad_dataset_split_idx'] = split_idx

    # Create dataloader
    train_loader = make_dataloader(bad_dataset_train, config)
    validation_loader = make_dataloader(bad_dataset_validation, config)
    anomaly_loader = make_dataloader(good_spectra_dataset, config)
    
    return train_loader, validation_loader, anomaly_loader

#%% Training cycle function

def train_anomaly_model(model, optimizer, loader_list, config, lr_scheduler = None):
    train_loader = loader_list[0]
    validation_loader = loader_list[1]
    anomaly_loader = loader_list[2]
    
    log_dict = {}
   
    for epoch in range(config['epochs']):
        # Compute loss (and eventually update weights)
        if config['use_as_autoencoder']:
            loss_function = torch.nn.MSELoss()
            train_loss      = epoch_ae(model, train_loader, config, True, loss_function, optimizer)
            validation_loss = epoch_ae(model, validation_loader, config, False, loss_function)
            anomaly_loss    = epoch_ae(model, anomaly_loader, config, False, loss_function)
        else:
            train_loss      = epoch_VAE(model, train_loader, config, True, optimizer)
            validation_loss = epoch_VAE(model, validation_loader, config, False)
            anomaly_loss    = epoch_VAE(model, anomaly_loader, config, False)
        
        # Save metric to load on wandb
        log_dict['learning_rate'] = optimizer.param_groups[0]['lr']
        if config['use_as_autoencoder']:
            log_dict, loss_string = divide_ae_loss([train_loss, validation_loss, anomaly_loss], log_dict)
        else:
            log_dict, loss_string = divide_VAE_loss([train_loss, validation_loss, anomaly_loss], log_dict)
        
        # Log data on wandb
        wandb.log(log_dict)
        
        # Update learning rate (if a scheduler is provided)
        if lr_scheduler is not None: lr_scheduler.step()
        
        if config['print_var']: 
            print("Epoch: {}".format(epoch))
            print(loss_string)
            
            
#%% Autoencoder
  
def epoch_ae(model, loader, config, is_train, loss_function, optimizer = None):    
    tot_loss = 0
    for batch in loader:
        x = batch.to(config['device'])
                
        if(is_train): # Executed during training
            # Zeros past gradients and forward step
            optimizer.zero_grad()
            recon_loss = loss_ae(x, model, loss_function)
            
            # Backward and optimization pass
            recon_loss.backward()
            optimizer.step()
        else: # Executed during testing
            with torch.no_grad(): # Deactivate the tracking of the gradient
                recon_loss = loss_ae(x, model, loss_function)
        
        # The multiplication serve to compute the average loss over the dataloader
        tot_loss += recon_loss * x.shape[0]
    
    # The division serve to compute the average loss over the dataloader
    tot_loss = tot_loss / len(loader.sampler)
    return tot_loss
          
      
def loss_ae(x, model, loss_function):
    # Divide data in the two spectra
    x1, x2, x = divide_spectra(x)
    # N.b. the new x contain x1 and x2 concatenated without the wavelength not used. So it is shorter than the original x
    
    # Forward pass
    x_r_1, x_r_2, z = model(x1, x2)
    
    # Loss computation
    x_r = torch.cat((x_r_1, x_r_2), -1)
    recon_loss = loss_function(x, x_r)
    
    return recon_loss


def divide_ae_loss(ae_loss_list, log_dict):
    tmp_dict = {}
    tmp_dict["reconstruction_loss_train"] = ae_loss_list[0]
    tmp_dict["reconstruction_loss_validation"] = ae_loss_list[1]
    tmp_dict["reconstruction_loss_anomaly"] = ae_loss_list[2]
    log_dict = {**log_dict, **tmp_dict}
    loss_string  = "\treconstruction_loss_train:      " + str(ae_loss_list[0].cpu().detach().numpy()) + "\n"
    loss_string += "\treconstruction_loss_validation: " + str(ae_loss_list[1].cpu().detach().numpy()) + "\n"
    loss_string += "\treconstruction_loss_anomaly:    " + str(ae_loss_list[2].cpu().detach().numpy())
    
    return log_dict, loss_string

#%% VAE

def epoch_VAE(model, loader, config, is_train, optimizer = None):
    recon_loss_total = 0
    KL_loss_total = 0
    for batch in loader:
        x = batch.to(config['device'])
        
        if is_train:
            optimizer.zero_grad()
            
            # Forward pass and compute VAE loss. 
            # The alpha and beta hyperparameters are used inside the loss function
            vae_loss, recon_loss, kl_loss = loss_VAE(x, model, config)
            
            # Backward pass (VAE loss contain the sum of recon and loss)
            vae_loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                vae_loss, recon_loss, kl_loss = loss_VAE(x, model, config)
          
        recon_loss_total = recon_loss * x.shape[0]
        KL_loss_total = kl_loss * x.shape[0]
        
    recon_loss_total /= len(loader.sampler)
    KL_loss_total /= len(loader.sampler)
    
    return [recon_loss_total, KL_loss_total]
    

def loss_VAE(x, model, config):
    if 'alpha' not in config: config['alpha'] = 1
    if 'beta' not in config: config['beta'] = 1
    
    # Divide the spectra
    x1, x2, x = divide_spectra(x)
    
    # Forward pass
    vae_output = model(x1, x2)
    
    # Average of the distribution of the reconstructed output
    x_r_1, x_r_2 = vae_output[0], vae_output[1]
    x_r = torch.cat((x_r_1, x_r_2), -1)
    
    # Variance of the distribution of the reconstructed output
    log_var_r_1, log_var_r_2 = vae_output[2], vae_output[3]
    log_var_r = torch.cat((log_var_r_1, log_var_r_2), -1)
    
    # Latent space mean and variance
    mu_z, log_var_z = vae_output[4], vae_output[5]
    
    # Compute loss
    vae_loss, recon_loss, kl_loss = VAE_loss(x, x_r, log_var_r, mu_z, log_var_z, config['alpha'], config['beta'])
    
    return vae_loss, recon_loss, kl_loss
    
    
def VAE_loss(x, x_r, log_var_r, mu_q, log_var_q, alpha = 1, beta = 1):
    """
    Loss of the VAE. 
    It return the reconstruction loss between x and x_r and the Kullback between a standard normal distribution and the ones defined by sigma and log_var
    It also return the sum of the two.
    The hyperparameter alpha multiply the reconstruction loss.
    The hyperparameter beta multiply the KL loss.
    """
    
    # Kullback-Leibler Divergence
    # N.b. Due to implementation reasons I pass to the function the STANDARD DEVIATION, i.e. the NON-SQUARED VALUE
    # When the variance is needed inside the function the sigmas are eventually squared
    
    sigma_p = torch.ones(log_var_q.shape).to(log_var_q.device) # Standard deviation of the target standard distribution
    mu_p = torch.zeros(mu_q.shape).to(mu_q.device) # Mean of the target gaussian distribution
    sigma_q = torch.sqrt(torch.exp(log_var_q)) # standard deviation obtained from the VAE
    kl_loss = KL_Loss(sigma_p, mu_p, sigma_q, mu_q).mean()

    # Reconstruction loss 
    sigma_r = torch.sqrt(torch.exp(log_var_r))
    recon_loss = VAE_recon_loss(x, x_r, sigma_r).mean()
    
    vae_loss = recon_loss * alpha + kl_loss * beta

    return vae_loss, recon_loss * alpha, kl_loss * beta


def KL_Loss(sigma_p, mu_p, sigma_q, mu_q):
    """
    General function for a KL loss with specified the paramters of two gaussian distributions p and q
    The parameter must be sigma (standard deviation) and mu (mean).
    The order of the parameter must be the following: sigma_p, mu_p, sigma_q, mu_q
    """
    
    tmp_el_1 = torch.log(sigma_q/sigma_p)
    
    tmp_el_2_num = torch.pow(sigma_q, 2) + torch.pow((mu_q - mu_p), 2)
    tmp_el_2_den = 2 * torch.pow(sigma_p, 2)
    tmp_el_2 = tmp_el_2_num / tmp_el_2_den
    
    kl_loss = - (tmp_el_1  - tmp_el_2 + 0.5)
    
    # P.s. The sigmas and mus have length equals to the hinner space dimension. So the final shape is [n_sample_in_batch, hidden_sapce_dimension]
    return kl_loss.sum(dim = 1)


def VAE_recon_loss(x, x_r, std_r):
    """
    Advance versione of the recontruction loss for the VAE when the output distribution is gaussian.
    Instead of the simple L2 loss we use the log-likelihood formula so we can also encode the variance in the output of the decoder.
    Input parameters:
      x = Original data
      x_r = mean of the reconstructed output
      std_r = standard deviation of the reconstructed output. 
    
    More info: 
    https://www.statlect.com/fundamentals-of-statistics/normal-distribution-maximum-likelihood
    https://arxiv.org/pdf/2006.13202.pdf
    """
    
    total_loss = 0
        
    # MSE Part (a variance per wavelength)
    mse_core = (torch.pow((x - x_r), 2)/(2 * torch.pow(std_r, 2))).sum(1) / x.shape[1]
    total_loss += mse_core 
    
    # Variance part
    # total_loss += x[0].shape[0] * torch.log(std_r).mean()
    
    return total_loss


def divide_VAE_loss(vae_loss_list, log_dict):
    tmp_dict = {}
    tmp_train_loss = vae_loss_list[0]
    tmp_validation_loss = vae_loss_list[1]
    tmp_anomaly_loss = vae_loss_list[2]
    
    # Add reconstruction loss
    tmp_dict["reconstruction_loss_train"] = tmp_train_loss[0]
    tmp_dict["reconstruction_loss_validation"] = tmp_validation_loss[0]
    tmp_dict["reconstruction_loss_anomaly"] = tmp_anomaly_loss[0]
    
    # Add KL loss
    tmp_dict["KL_loss_train"] = tmp_train_loss[1]
    tmp_dict["KL_loss_validation"] = tmp_validation_loss[1]
    tmp_dict["KL_loss_anomaly"] = tmp_anomaly_loss[1]
    
    # Add loss to log dict
    log_dict = {**log_dict, **tmp_dict}
    
    # String to display
    loss_string =  "\treconstruction_loss_train:      " + str(tmp_train_loss[0].cpu().detach().numpy()) + "\n"
    loss_string += "\treconstruction_loss_validation: " + str(tmp_validation_loss[0].cpu().detach().numpy()) + "\n"
    loss_string += "\treconstruction_loss_anomaly:    " + str(tmp_anomaly_loss[0].cpu().detach().numpy()) + "\n\n"
    loss_string += "\tKL_loss_train:      " + str(tmp_train_loss[1].cpu().detach().numpy()) + "\n"
    loss_string += "\tKL_loss_validation: " + str(tmp_validation_loss[1].cpu().detach().numpy()) + "\n"
    loss_string += "\tKL_loss_anomaly:    " + str(tmp_anomaly_loss[1].cpu().detach().numpy())
    
    return log_dict, loss_string

#%% Other functions

def divide_spectra(x):
    length_mems_1 = 300
    length_mems_2 = 400
    
    # Divide data in the two spectra
    x1 = x[..., 0:length_mems_1]
    x2 = x[..., (- 1 - length_mems_2):-1]
    x = torch.cat((x1,x2), -1)
    
    return x1, x2, x

def compute_accuracy(model, loader, device):
    """
    Compute the accuracy (i.e. the percentage of correctly classified exampled) for the sequence embedding classifier
    """
    
    n_example = 0
    tot_correct = 0
    
    for batch in loader:
        x = batch[0].to(device)
        y_true = batch[1].to(device)
        
        y = model(x)
        y = torch.argmax(y, 1)
        
        tot_correct += torch.sum(y == y_true)
        n_example += x.shape[0]
        
    accuracy = tot_correct / n_example
    
    return accuracy