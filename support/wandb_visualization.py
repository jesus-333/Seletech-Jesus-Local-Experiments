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
from support.wandb_training_V2 import loss_ae, loss_VAE

#%%

def bar_loss_wandb_V1(project_name, dataloader_list, config):
    with wandb.init(project = project_name, job_type = "plot", config = config) as run:
        
        # Load model
        model, model_config = load_model_from_artifact_inside_run(run, config['artifact_name'], config['version'], config['model_name'])
        config['use_as_autoencoder'] = model_config['use_as_autoencoder']
        model.eval()
        
        # Compute loss
        loss_list = []
        for dataloader in dataloader_list:
            tmp_loss = compute_loss(dataloader, model, config)
            loss_list.append(tmp_loss)
        
        # Plot the loss in the bar chart
        fig, ax = plot_bar_loss(loss_list, config)
        
        # Log the plot
        wandb.log({"Bar Error Chart": fig})
                 
        # Save the plot in the artifact
        plot_description = "Error bar chart of a trained model"
        path = 'TMP_File/error_bar_chart'
        plot_artifact = wandb.Artifact("Error_bar_plot", type = "plot", description = plot_description, metadata = dict(config))
        save_plot_and_add_to_artifact(fig, path, 'png', plot_artifact)
        save_plot_and_add_to_artifact(fig, path, 'eps', plot_artifact)
        save_plot_and_add_to_artifact(fig, path, 'tex', plot_artifact)
        run.log_artifact(plot_artifact)
        
        return fig, ax

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
            vae_loss, recon_loss, kl_loss = loss_VAE(x, model)
        
        tot_loss += recon_loss * x.shape[0]
        
    tot_loss = tot_loss / len(dataloader.dataset)
    
    return tot_loss.cpu().detach()


def plot_bar_loss(loss_list, config):
    if len(loss_list) != len(config['dataset_labels']): raise ValueError("The number of label must be equals to the number of class (dataloader)")
    
    fig, ax = plt.subplots(1,1, figsize = config['figsize'])
    
    ax.bar(config['dataset_labels'], loss_list, color = config['colors'])
    
    ax.set_ylabel(config['ylabel'])
    
    plt.rcParams.update({'font.size': config['fontsize']})
    plt.tight_layout()
    
    return fig, ax


def save_plot_and_add_to_artifact(fig, path, file_type, artifact):
    if file_type == 'tex':
        try:
            import tikzplotlib
            text_file = open("{}.{}".format(path, file_type), "w")
            n = text_file.write(tikzplotlib.get_tikz_code(fig))
            text_file.close()
        except:
            raise ImportError("tikzplotlib not installed. To export plot in tikz install the package.")
    else:
        fig.savefig("{}.{}".format(path, file_type), format = file_type)
        
    artifact.add_file("{}.{}".format(path, file_type))
    wandb.save()