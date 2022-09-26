#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 14:43:02 2022

@author: Alberto Zancanaro (jesus)
@organization: University of Padua
"""

#%%

import wandb

import numpy as np
import matplotlib.pyplot as plt
import torch

from support.VAE import SpectraVAE_Double_Mems
from support.wandb_init_V2 import load_model_from_artifact_inside_run, add_model_to_artifact
from support.wandb_training_V1 import train_model_wandb

#%%

def train_and_log_model(project_name, loader_list, config):
    with wandb.init(project = project_name, job_type = "train", config = config) as run:
        config = wandb.config
        
        train_loader = loader_list[0]
        validation_loader = loader_list[1]
        anomaly_loader = loader_list[2]
        
        # Load model from artifacts
        model, model_config = load_model_from_artifact_inside_run(run, config['model_artifact_name'],
                                                    version = 'latest', model_name = 'untrained.pth')
        config['use_as_autoencoder'] = model_config['use_as_autoencoder']
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr = config['lr'], 
                                      weight_decay = config['optimizer_weight_decay'])
        
        # Setup lr scheduler
        if config['use_scheduler'] == True:
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = config['gamma'])
        else:
            lr_scheduler = None
            
        if config['print_var']: print("Model trained on: ".format(config['device']))
        
        # Train model
        model, model_config = train_model_wandb(model, optimizer, 
                                  train_loader, validation_loader,anomaly_loader, 
                                  config, lr_scheduler)
        
        # Save model after training
        model_artifact_name = config['model_artifact_name'] + '_trained'
        model_artifact = wandb.Artifact(model_artifact_name, type = "model",
                                        description = "Trained {} model".format_map(config['model_artifact_name']),
                                        metadata = dict(model_config))
        add_model_to_artifact(model, model_artifact)
        run.log_artifact(model_artifact)
        
        return model
            

