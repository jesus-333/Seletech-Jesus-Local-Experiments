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

from support.wandb_init_V1 import make_dataloader
from support.datasets import PytorchDatasetPlantSpectra_V1
from support.wandb_init_V2 import load_VAE_trained_model_from_artifact_inside_run, load_dataset_from_artifact_inside_run
from support.wandb_training_VAE import loss_ae, loss_VAE

#%%

def bar_loss_wandb_V1(project_name, config):
    with wandb.init(project = project_name, job_type = "plot", config = config) as run:
        if 'device' not in config: config['device'] = 'cpu'
        
        # Load model
        model, model_config, idx_dict = load_VAE_trained_model_from_artifact_inside_run(config, run)
        config['use_as_autoencoder'] = model_config['use_as_autoencoder']
        config['dataset_config']['use_cnn'] = model_config['use_cnn']
        model.eval()
        model.to(config['device'])
        
        # Load data from dataset artifact
        data = load_dataset_from_artifact_inside_run(config['dataset_config'], run)
        spectra = data[0]
        
        # Create dataset
        good_spectra_dataset = PytorchDatasetPlantSpectra_V1(spectra[idx_dict['good_idx'], :], used_in_cnn = config['dataset_config']['use_cnn'])
        bad_spectra_dataset = PytorchDatasetPlantSpectra_V1(spectra[idx_dict['bad_idx'], :], used_in_cnn = config['dataset_config']['use_cnn'])
        
        # Divide bad(dry) dataset in train/test/validation
        bad_dataset_train = bad_spectra_dataset[idx_dict['bad_dataset_split_idx'][0]]
        bad_dataset_test = bad_spectra_dataset[idx_dict['bad_dataset_split_idx'][1]]
        bad_dataset_validation = bad_spectra_dataset[idx_dict['bad_dataset_split_idx'][2]]
        
        # Create dataloader
        train_loader = make_dataloader(bad_dataset_train, config)
        validation_loader = make_dataloader(bad_dataset_validation, config)
        anomaly_loader = make_dataloader(good_spectra_dataset, config)
        dataloader_list = [train_loader, validation_loader, anomaly_loader]
    
        # Compute loss
        avg_loss_list = []
        std_loss_list = []
        for dataloader in dataloader_list:
            avg_loss, std_loss = compute_avg_loss(dataloader, model, config)
            avg_loss_list.append(avg_loss)
            std_loss_list.append(std_loss)
        
        # Plot the loss in the bar chart
        fig, ax = plot_bar_loss(avg_loss_list, std_loss_list, config)
        
        # Save the plot in the artifact
        plot_description = "Error bar chart of a trained model"
        path = 'TMP_File/error_bar_chart'
        plot_artifact = wandb.Artifact("Error_bar_plot", type = "plot", description = plot_description, metadata = dict(config))
        save_plot_and_add_to_artifact(fig, path, 'png', plot_artifact)
        save_plot_and_add_to_artifact(fig, path, 'eps', plot_artifact)
        save_plot_and_add_to_artifact(fig, path, 'tex', plot_artifact)
        run.log_artifact(plot_artifact)
        
        # Log the plot
        # wandb.log({"Bar Error Chart": fig})
        
        return fig, ax

def compute_avg_loss(dataloader, model, config):
    """
    Compute the average loss of the data inside a given dataloader
    """
    
    loss_function = torch.nn.MSELoss()
    
    tot_loss = 0
    loss_list = []
    for batch in dataloader:
        # Move data to device
        x = batch.to(config['device'])
        
        # Compute loss
        if(config['use_as_autoencoder']):
            recon_loss = loss_ae(x, model, loss_function)
        else:
            vae_loss, recon_loss, kl_loss = loss_VAE(x, model, config)
        
        tot_loss += recon_loss * x.shape[0]
        loss_list.append(recon_loss)
    
    # Compute average and std of the loss
    avg_loss = tot_loss / len(dataloader.dataset)
    std_loss = torch.stack(loss_list).view(-1).std()
    
    return avg_loss.cpu().detach(), std_loss.cpu().detach()


def plot_bar_loss(loss_list, std_loss_list, config):
    if len(loss_list) != len(config['dataset_labels']): raise ValueError("The number of label must be equals to the number of class (dataloader)")
    
    fig, ax = plt.subplots(1,1, figsize = config['figsize'])
    
    if config['add_std_bar']:
        ax.bar(config['dataset_labels'], loss_list, yerr = std_loss_list, color = config['colors'], error_kw = config['error_kw'])
    else:
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