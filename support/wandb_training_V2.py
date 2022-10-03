#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 14:43:02 2022

@author: Alberto Zancanaro (jesus)
@organization: University of Padua
"""

#%% Imports

import wandb

import numpy as np
import matplotlib.pyplot as plt
import torch

from support.VAE import SpectraVAE_Double_Mems
from support.wandb_init_V2 import load_model_from_artifact_inside_run, add_model_to_artifact
from support.wandb_training_V1 import train_model_wandb

#%% Principal function

def train_and_log_model(project_name, loader_list, config):
    with wandb.init(project = project_name, job_type = "train", config = config) as run:
        config = wandb.config
        
        # Load model from artifacts
        model, model_config = load_model_from_artifact_inside_run(run, config['model_artifact_name'],
                                                    version = config['version'], model_name = 'untrained.pth')
        config['use_as_autoencoder'] = model_config['use_as_autoencoder']
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr = config['lr'], 
                                      weight_decay = config['optimizer_weight_decay'])
        
        # Setup lr scheduler
        if config['use_scheduler'] == True:
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = config['gamma'])
        else:
            lr_scheduler = None
            
        if config['print_var']: print("Model trained on: {}".format(config['device']))
        
        # Train model
        model.to(config['device'])
        wandb.watch(model, log = "all", log_freq = config['log_freq'])
        if config['use_as_autoencoder']: train_model_ae(model, optimizer, loader_list, config, lr_scheduler)
        
        # Save model after training
        model_artifact_name = config['model_artifact_name'] + '_trained'
        model_artifact = wandb.Artifact(model_artifact_name, type = "model",
                                        description = "Trained {} model".format(config['model_artifact_name']),
                                        metadata = dict(model_config))
        add_model_to_artifact(model, model_artifact)
        run.log_artifact(model_artifact)
        
        return model
            

#%% Training autoencoder

def train_model_ae(model, optimizer, loader_list, config, lr_scheduler = None):
    train_loader = loader_list[0]
    validation_loader = loader_list[1]
    anomaly_loader = loader_list[2]
    
    log_dict = {}
    loss_function = torch.nn.MSELoss()
    for epoch in range(config['epochs']):
        # Compute loss (and eventually update weights)
        train_loss      = epoch_ae(model, train_loader, config, True, loss_function, optimizer)
        validation_loss = epoch_ae(model, validation_loader, config, False, loss_function)
        anomaly_loss    = epoch_ae(model, anomaly_loader, config, False, loss_function)
        
        # Save metric to load on wandb
        log_dict['learning_rate'] = optimizer.param_groups[0]['lr']
        log_dict, loss_string = divide_ae_loss([train_loss, validation_loss, anomaly_loss], log_dict)
        
        # Log data on wandb
        wandb.log(log_dict)
        
        # Update learning rate (if a scheduler is provided)
        if lr_scheduler is not None: lr_scheduler.step()
        
        if config['print_var']: 
            print("Epoch: {}".format(epoch))
            print(loss_string)
    
    
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
    length_mems_1 = 300
    length_mems_2 = 400
    
    # Divide data in the two spectra
    x1 = x[..., 0:length_mems_1]
    x2 = x[..., (- 1 - length_mems_2):-1]
    x = torch.cat((x1,x2), -1)
    
    x_r_1, x_r_2, z = model(x1, x2)
    x_r = torch.cat((x_r_1, x_r_2), -1)
    recon_loss = loss_function(x, x_r)
    
    return recon_loss


def divide_ae_loss(ae_loss_list, log_dict):
    tmp_dict = {}
    tmp_dict["reconstruction_loss_train"] = ae_loss_list[0]
    tmp_dict["reconstruction_loss_validation"] = ae_loss_list[1]
    tmp_dict["reconstruction_loss_anomaly"] = ae_loss_list[2]
    log_dict = {**log_dict, **tmp_dict}
    loss_string = "\treconstruction_loss_train:       " + str(ae_loss_list[0].cpu().detach().numpy()) + "\n"
    loss_string += "\treconstruction_loss_validation: " + str(ae_loss_list[1].cpu().detach().numpy()) + "\n"
    loss_string += "\treconstruction_loss_anomaly:    " + str( ae_loss_list[2].cpu().detach().numpy())
    
    return log_dict, loss_string

#%% Training VAE

def train_model_VAE(model, optimizer, loader_list, config, lr_scheduler = None):
    train_loader = loader_list[0]
    validation_loader = loader_list[1]
    anomaly_loader = loader_list[2]
    
    log_dict = {}
    
    for epoch in range(config['epochs']):
        pass
    
    
def loss_vae(x, model):
    pass