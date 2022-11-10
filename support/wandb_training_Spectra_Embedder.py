"""
Created on Tue Nov  8 17:00:17 2022

@author: Alberto Zancanaro (jesus)
@organization: University of Padua

Contains the function to train the spectra embedder (SE)
"""

#%% Imports

import numpy as np
import torch
import wandb

from support.wandb_init_V2 import get_run_name
from support.wandb_init_V1 import make_dataloader
from support.wandb_init_V2 import load_dataset_from_artifact_inside_run, split_data
from support.wandb_init_V2 import load_untrained_model_from_artifact_inside_run, add_model_to_artifact
from support.datasets import SpectraNLPDataset

#%% Principal function

def train_and_log_SE_model(project_name, config):
    # Get the name for the actual run
    if "skipGram" in config['model_artifact_name']:  run_name = get_run_name('train-SE-skipgram-embedding')
    elif "CBOW" in config['model_artifact_name']:  run_name = get_run_name('train-SE-CBOW-embedding')
    else: raise ValueError("Problem with the type of model you want to build")

    
    # if config['dataset_config']['return_other_sensor_data'] == False and config['train_with_info_data']:
    #     raise ValueError("To use the other sensor data during the training you have to load them. The parameter return_other_sensor_data must be set to True")
    
    with wandb.init(project = project_name, job_type = "train", config = config, name = run_name) as run:
        config = wandb.config
          
        # Load model from artifacts
        model, model_config = load_untrained_model_from_artifact_inside_run(run, config['model_artifact_name'],
                                                    version = config['version'], model_name = 'untrained.pth')
            
        # Setup optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr = config['lr'], 
                                      weight_decay = config['optimizer_weight_decay'])
        
        # Setup lr scheduler
        if config['use_scheduler'] == True:
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = config['gamma'])
        else:
            lr_scheduler = None
            
        # Setup artifact to save model
        model_artifact_name = config['model_artifact_name'] + '_trained'
        metadata = dict(training_config = dict(config), model_config = model_config)
        model_artifact = wandb.Artifact(model_artifact_name, type = "model",
                                        description = "Trained {}:{} model".format(config['model_artifact_name'], config['version']),
                                        metadata = metadata)
        
        # Print the training device
        if config['print_var']: print("Model trained on: {}".format(config['device']))
        
        # Setup the dataloader
        loader= load_loader(config, run)
        if config['print_var']: print("Dataset loaded")
        
        # Train model
        wandb.watch(model, log = "all", log_freq = config['log_freq'])
        model.to(config['device'])
        train_spectra_embeddeding_model(model, optimizer, loader, model_artifact, config, lr_scheduler)
        
        # Save model after training
        add_model_to_artifact(model, model_artifact, "TMP_File/model_END.pth")
        run.log_artifact(model_artifact)
        
    return model

#%% Load data for training

def load_loader(config, run):
    # Load data from dataset artifact and get the spectra
    data = load_dataset_from_artifact_inside_run(config['dataset_config'], run)
    spectra = data[0]
    
    # Create dataset and dataloader
    dataset = SpectraNLPDataset(spectra, config['dataset_config'])
    loader = make_dataloader(dataset, config)
    
    return loader

#%% 

def train_spectra_embeddeding_model(model, optimizer, loader, model_artifact, config, lr_scheduler = None):
    # Parameter used to save the model every x epoch
    if 'epoch_to_save_model' not in config: config['epoch_to_save_model'] = 1
    
    log_dict = {}
    # Check the type of model
    loss_function = torch.nn.MSELoss()
    
    for epoch in range(config['epochs']):
        # Save lr
        if config['use_scheduler']:
            log_dict['learning_rate'] = optimizer.param_groups[0]['lr']
        
        # Compute loss (and eventually update weights)
        if 'clf' in str(type(model)).lower():
            # Advance epoch
            train_loss, train_acc = epoch_sequence_embeddeding_clf(model, train_loader, config, True, loss_function, optimizer)
            validation_loss, validation_acc = epoch_sequence_embeddeding_clf(model, validation_loader, config, False, loss_function)
            
            # Update log dict
            log_dict, loss_string = update_clf_log_dict([train_loss, validation_loss, train_acc, validation_acc], log_dict)
        elif 'autoencoder' in str(type(model)).lower():
            # Advance epoch
            train_loss = epoch_sequence_embeddeding_autoencoder(model, train_loader, config, True, loss_function, optimizer)
            validation_loss = epoch_sequence_embeddeding_autoencoder(model, validation_loader, config, False, loss_function)
            
            log_dict, loss_string = update_autoencoder_log_dict([train_loss, validation_loss], log_dict)
        else:
            raise ValueError("Error with sequence embedder model type. Must be classifier or autoencoder")
        
        # Save the model after the epoch
        # N.b. When the variable epoch is 0 the model is trained for an epoch when arrive at this instructions.
        if (epoch + 1) % config['epoch_to_save_model'] == 0:
            add_model_to_artifact(model, model_artifact, "TMP_File/model_{}.pth".format(epoch + 1))
        
        # Log data on wandb
        wandb.log(log_dict)
        
        # Update learning rate (if a scheduler is provided)
        if lr_scheduler is not None: lr_scheduler.step()
        
        if config['print_var']: 
            print("Epoch: {}".format(epoch))
            print(loss_string)

    