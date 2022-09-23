#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 15:16:29 2022

@author: Alberto Zancanaro (jesus)
@organization: University of Padua
"""

#%%

import numpy as np
import torch
import wandb

from support.VAE import SpectraVAE_Double_Mems, AttentionVAE
from support.VAE_Conv import SpectraVAE_Double_Mems_Conv

#%%

def build_and_log_model(config):
    if config['neurons_per_layer'][1] / config['neurons_per_layer'][0] != 2:
        raise ValueError("FOR NOW, the number of neurons of the second hidden layer must be 2 times the number of neurons in the first inner layer")
        
    with wandb.init(project = config['project_name'], job_type = "model_creation", config = config) as run:
        config = wandb.config
        
        model, model_description = build_model(config)
        
        # Create the artifacts
        metadata = dict(config)
        model_artifact = wandb.Artifact("SpectraVAE", type = "model", description = model_description, metadata = metadata)

        # Save the model and log it on wandb
        model_name = "TMP_file/untrained.pth"
        torch.save(model.state_dict(), model_name)
        model_artifact.add_file(model_name)
        wandb.save(model_name)
        
        
def build_model(config):
    # Create the model
    if config['use_cnn']: # Convolutional VAE 
        model = SpectraVAE_Double_Mems_Conv(config['length_mems_1'], config['length_mems_2'], config['hidden_space_dimension'], 
                                            use_as_autoencoder = config['use_as_autoencoder'], use_bias = config['use_bias'],
                                            print_var = config['print_var'])
        model_description = "Untrained VAE model. Convolutional version."
    else:
        if config['use_attention']: # Feed-Forward VAE with attention
            model = AttentionVAE(config['length_mems_1'], config['length_mems_2'], 
                               config['hidden_space_dimension'], config['embedding_size'],
                               print_var = config['print_var'], use_as_autoencoder = config['use_as_autoencoder'] )
            model_description = "Untrained VAE model. Fully-connected version + Attention."
        else: # Feed-Forward VAE without attention
            model = SpectraVAE_Double_Mems(config['length_mems_1'], config['length_mems_2'], config['hidden_space_dimension'], 
                                         use_as_autoencoder = config['use_as_autoencoder'], use_bias = config['use_bias'],
                                         print_var = config['print_var'])
            model_description = "Untrained VAE model. Fully-connected version."
            
    return model, model_description