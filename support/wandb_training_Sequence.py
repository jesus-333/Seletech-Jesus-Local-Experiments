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

from support.wandb_init_V2 import load_model_from_artifact_inside_run, load_dataset_from_artifact_inside_run, add_model_to_artifact
from support.preprocess import choose_spectra_based_on_water_V1
from support.dataset import PytorchDatasetPlantSpectra_V1

#%% Principal function

def train_and_log_SE_model(project_name, config):
    with wandb.init(project = project_name, job_type = "train", config = config) as run:
        config = wandb.config
          
        # Load model from artifacts
        model, model_config = load_model_from_artifact_inside_run(run, config['model_artifact_name'],
                                                    version = config['version'], model_name = 'untrained.pth')
        
        # IF load the VAE/AE model check if it is used as autoencoder
        if "VAE" in str(type(model)): config['use_as_autoencoder'] = model_config['use_as_autoencoder']
        
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
        metadata = dict(training_config = config, model_config = model_config)
        model_artifact = wandb.Artifact(model_artifact_name, type = "model",
                                        description = "Trained {}:{} model".format(config['model_artifact_name'], config['version']),
                                        metadata = metadata)
        
        # Print the training device
        if config['print_var']: print("Model trained on: {}".format(config['device']))
        
        # Setup the dataloader
        loader_list = load_loader(config)
        
        # Train model
        model.to(config['device'])
        train_sequence_embeddeding_clf_model(model, optimizer, loader_list, config, lr_scheduler)

        add_model_to_artifact(model, model_artifact)
        run.log_artifact(model_artifact)
        
        return model

#%% Load data for training

def load_loader(config, run):
    pass

#%% Training cycle function

def train_sequence_embeddeding_clf_model(model, optimizer, loader_list, config, lr_scheduler = None):
    train_loader = loader_list[0]
    validation_loader = loader_list[1]
    
    log_dict = {}
    loss_function = torch.nn.NLLLoss()
   
    for epoch in range(config['epochs']):
        # Save lr 
        log_dict['learning_rate'] = optimizer.param_groups[0]['lr']
        
        # Compute loss (and eventually update weights)
        if 'clf' in str(type(model)):
            # Advance epoch
            train_loss, train_acc = epoch_sequence_embeddeding_clf(model, train_loader, config, True, loss_function, optimizer)
            validation_loss, validation_acc = epoch_sequence_embeddeding_clf(model, validation_loader, config, False, loss_function)
            
            # Update log dict
            log_dict, loss_string = update_clf_log_dict([train_loss, validation_loss, train_acc, validation_acc], log_dict)
        elif 'autoencoder' in str(type(model)):
            pass
        else:
            raise ValueError("Error with sequence embedder model type. Must be classifier or autoencoder")
        
        
        # Log data on wandb
        wandb.log(log_dict)
        
        # Update learning rate (if a scheduler is provided)
        if lr_scheduler is not None: lr_scheduler.step()
        
        if config['print_var']: 
            print("Epoch: {}".format(epoch))
            print(loss_string)


#%% Sequence Embeddeding Clf

def epoch_sequence_embeddeding_clf(model, loader, config, is_train, loss_function, optimizer = None):    
    tot_loss = 0
    for batch in loader:
        x = batch[0].to(config['device'])
        y_true = batch[1].to(config['device'])
                
        if(is_train): # Executed during training
            # Zeros past gradients and forward step
            optimizer.zero_grad()
            
            y = model(x)
            clf_loss = loss_function(y, y_true)
            
            # Backward and optimization pass
            clf_loss.backward()
            optimizer.step()
        else: # Executed during testing
            with torch.no_grad(): # Deactivate the tracking of the gradient
                y = model(x)
                clf_loss = loss_function(y, y_true)
        
        # The multiplication serve to compute the average loss over the dataloader
        tot_loss += clf_loss * x.shape[0]
    
    # The division serve to compute the average loss over the dataloader
    tot_loss = tot_loss / len(loader.sampler)
    
    # Compute accuracy at the end of the epoch
    accuracy = compute_accuracy(model, loader, config['device'])
    
    return tot_loss, accuracy


def update_clf_log_dict(clf_loss_list, log_dict):
    tmp_dict = {}
    tmp_dict["clf_loss_train"] = clf_loss_list[0]
    tmp_dict["clf_loss_validation"] = clf_loss_list[1]
    tmp_dict["clf_acc_train"] = clf_loss_list[2]
    tmp_dict["clf_acc_validation"] = clf_loss_list[3]
    log_dict = {**log_dict, **tmp_dict}
    loss_string  = "\tclf_loss_train:      " + str(clf_loss_list[0].cpu().detach().numpy()) + "\n"
    loss_string += "\tclf_loss_validation: " + str(clf_loss_list[1].cpu().detach().numpy()) + "\n"
    loss_string += "\tclf_acc_train:       " + str(clf_loss_list[2].cpu().detach().numpy()) + "\n"
    loss_string += "\tclf_acc_validation:  " + str(clf_loss_list[3].cpu().detach().numpy())
    
    return log_dict, loss_string

#%% Sequence embedding autoencoder

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