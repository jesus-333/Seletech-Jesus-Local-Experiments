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
from support.wandb_init_V2 import load_trained_model_from_artifact_inside_run, load_dataset_from_artifact_inside_run
from support.wandb_training_VAE import loss_ae, loss_VAE
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

#%% Error bar for VAE

def bar_loss_wandb_V1(project_name, config):
    with wandb.init(project = project_name, job_type = "plot", config = config) as run:
        if 'device' not in config: config['device'] = 'cpu'
        
        # Load model
        model, model_config, idx_dict = load_trained_model_from_artifact_inside_run(config, run)
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
    
    
#%% Spectra/Sequence embedding

def plot_spectra_embedding(embedding, config):
    fig, ax = plt.subplots(figsize = config['figsize'])
    sc = ax.scatter(embedding[:,0], embedding[:,1], c = config['color'], s = config['s'],  cmap = config['cmap'])
    
    if 'xlim' in config: ax.set_xlim(config['xlim'])
    if 'ylim' in config: ax.set_xlim(config['ylim'])
    
    plt.colorbar(sc)
    plt.show()

    return fig, ax

def compute_embedding(embedder, loader, config):
    """
    Compute the embedding (both for sequence and for spectra)
    You need to pass a pytorch dataloader. Used if you have a lot of data
    """
    embedder.to(config['device'])
    embedding_list = []
    
    for batch in loader:
        # Move data to device
        x = batch[0].to(config['device'])
        
        # Compute the embedding
        if config['input_type'] == 'spectra_embedding':
            # TODO Implement CBOW version. For now works only for skipGram
            if 'CBOW' in str(type(embedder)): raise ValueError("The embedding for now works only for skipGram")
            tmp_emb = embedder(x)
        elif config['input_type'] == 'sequence_embedding':
            # out is all the output of the LSTM  (see PyTorch LSTM documentation)
            # h is the embedding
            # c is the last state of the LSTM encoder
            out, h, c = embedder(x)
            
            tmp_emb = h
            
        # Save the results for the batch
        embedding_list.append(tmp_emb.detach().cpu().squeeze())
    
    # Convert embedding list in a single numpy array
    embedding = torch.cat(embedding_list).numpy()
    
    return embedding

def fast_compute_embedding(embedder, data, device = 'cpu'):
    embedder.to(device)
    embedding = embedder(data.to(device).float())

    return embedding.detach().cpu()

def reduce_dimension(x, final_dimension, method):
    if method == 'tsne':
        x = TSNE(n_components = final_dimension, learning_rate='auto', init='random').fit_transform(x)
    elif method == 'pca':
        x = PCA(n_components = final_dimension).fit_transform(x)
    else: 
        raise ValueError("The methods must have value pca or tsne")
        
    return x


def plot_evolution(spectra, embedder, config):
    for i in range(len(config['idx_list'])):
        idx = config['idx_list'][i]

        # Load the path for the current epoch
        model_path = config['path_weight'] + str(idx) + '.pth'
        embedder.load_state_dict(torch.load(model_path, map_location = torch.device('cpu')))
        embedder.to(config['device'])
        embedder.eval()
        
        # Compute the embedding
        with torch.no_grad():
            embedding = embedder(spectra.to(config['device']).float()).cpu().numpy()
        
        # Reduce the dimension to plot
        if embedding.shape[1] > 2: embedding = reduce_dimension(embedding, 2, config['dimensionaly_reduction_method'])

        plot_spectra_embedding(embedding, config)

#%% End file
