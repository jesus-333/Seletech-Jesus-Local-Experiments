#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 13:43:33 2022

@author: Alberto Zancanaro (jesus)
@organization: University of Padua
"""


#%% Imports 

import numpy as np
import matplotlib.pyplot as plt
import torch
import wandb

from support.wandb_init_V2 import load_model_from_artifact_inside_run
from support.wandb_training_V2 import loss_ae, loss_vae

#%%

def bar_loss_wandb_V1(project_name, dataloader_list, config):
    with wandb.init(project = project_name, job_type = "plot", config = config) as run:
        
        # Load model
        model, model_config = load_model_from_artifact_inside_run(run, config['artifact_name'], config['version'], config['model_name'])
        config['use_as_autoencoder'] = model_config['use_as_autoencoder']
        
        # Compute loss
        loss_list = []
        for dataloader in dataloader_list:
            tmp_loss = compute_loss(dataloader, config)
            loss_list.append(tmp_loss)
        
        # Plot the loss in the bar chart
        fig, ax = plot_bar_loss(loss_list, config)
        
        # Log th plot
        wandb.log({"Bar Error Chart": fig})
        
        # Show the plot
        plt.show()

            
def compute_loss(dataloader, model, config):
    """
    Compute the loss of the data inside a given dataloader
    """
    
    loss_function = torch.nn.MSELoss()
    
    tot_loss = 0
    for batch in dataloader:
        # Move data to device
        x = batch.to(config['device'])
        
        # Compute loss
        if(config['use_as_autoencoder']):
            recon_loss = loss_ae(x, model, loss_function)
        else:
            recon_loss = loss_vae(x, model)
        
        tot_loss += recon_loss * x.shape[0]
        
    tot_loss = tot_loss / len(dataloader.dataset)
    
    return tot_loss


def plot_bar_loss(loss_list, config):
    if len(loss_list) != len(config['labels']): raise ValueError("The number of label must be equals to the number of class (dataloader)")
    
    fig, ax = plt.subplots(1,1, figsize = config['figsize'])
    
    ax.bar(config['labels'], loss_list, color = config['color'])
    
    ax.set_xlabel(config['xlabel'])
    ax.set_ylabel(config['ylabel'])
    
    plt.rcParams.update({'font.size': config['fontsize']})
    plt.tight_layout()
    
    return fig, ax