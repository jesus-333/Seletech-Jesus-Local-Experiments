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
import pickle

from support.wandb_init_V1 import make_dataloader
from support.wandb_init_V2 import load_dataset_from_artifact_inside_run, split_data
from support.wandb_init_V2 import load_untrained_model_from_artifact_inside_run, add_model_to_artifact
from support.wandb_init_V2 import get_run_name
from support.datasets import SpectraSequenceDataset

#%% Principal function

def train_and_log_SE_model(project_name, config):
    # Get the name for the actual run
    if "SequenceEmbedder_clf" in config['model_artifact_name']:  run_name = get_run_name('train-SE-clf-embedding')
    elif "SequenceEmbedder_AE" in config['model_artifact_name']: run_name = get_run_name('train-SE-AE-embedding')
    else: raise ValueError("Problem with the type of model you want to load")
    
    if config['dataset_config']['return_other_sensor_data'] == False and config['train_with_info_data']:
        raise ValueError("To use the other sensor data during the training you have to load them. The parameter return_other_sensor_data must be set to True")
    
    
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
        metadata = dict(training_config = config, model_config = model_config)
        model_artifact = wandb.Artifact(model_artifact_name, type = "model",
                                        description = "Trained {}:{} model".format(config['model_artifact_name'], config['version']),
                                        metadata = metadata)
        
        # Print the training device
        if config['print_var']: print("Model trained on: {}".format(config['device']))
        
        # Setup the dataloader
        loader_list, idx_dict = load_loader(config, run)
        save_idx_to_artifact(idx_dict, model_artifact, run)
        if config['print_var']: print("Dataset loaded")
        
        # Train model
        wandb.watch(model, log = "all", log_freq = config['log_freq'])
        model.to(config['device'])
        train_sequence_embeddeding_model(model, optimizer, loader_list, model_artifact, config, lr_scheduler)
        
        # Save model after training
        add_model_to_artifact(model, model_artifact, "model_END.pth")
        run.log_artifact(model_artifact)
        
    return model


#%% Load data for training

def load_loader(config, run):
    # Load data from dataset artifact and get the spectra
    data = load_dataset_from_artifact_inside_run(config['dataset_config'], run)
    spectra = data[0]
    
    # Split the data
    train_idx, test_idx, validation_idx, split_idx = split_data(spectra, config)
    
    if config['train_with_info_data']: 
        info_train = data[4][train_idx]
        info_test = data[4][test_idx]
        info_validation = data[4][validation_idx]
    else:
        info_train = info_test = info_validation = None
    
    # Create train, test and validation dataset
    dataset_train = SpectraSequenceDataset(spectra[train_idx], config['dataset_config'], info_train)
    dataset_test = SpectraSequenceDataset(spectra[test_idx], config['dataset_config'], info_test)
    dataset_validation = SpectraSequenceDataset(spectra[validation_idx], config['dataset_config'], info_validation)      

    # Create dataloader
    train_loader = make_dataloader(dataset_train, config)
    validation_loader = make_dataloader(dataset_validation, config)
    test_loader = make_dataloader(dataset_test, config)
    
    # Save the idx used to divide the dataset between normal(dry/bad) and anomaly (good/water) (The dry spectra are more numerous)
    # Save how bad dataset is split between train/test/validation
    idx_dict = {}
    idx_dict['dataset_split_idx'] = split_idx
    
    loader_list = [train_loader, validation_loader, test_loader]
    
    return loader_list, idx_dict
    
    

#%% Training cycle function

def train_sequence_embeddeding_model(model, optimizer, loader_list, model_artifact, config, lr_scheduler = None):
    train_loader = loader_list[0]
    validation_loader = loader_list[1]
    
    # Parameter used to save the model every x epoch
    if 'epoch_to_save_model' not in config: config['epoch_to_save_model'] = 1
    
    log_dict = {}
    # Check the type of model
    if 'clf' in str(type(model)).lower(): loss_function = torch.nn.NLLLoss()
    elif 'autoencoder' in str(type(model)).lower(): loss_function = torch.nn.MSELoss()
    else: raise ValueError("Error with sequence embedder model type. Must be classifier or autoencoder")
   
    for epoch in range(config['epochs']):
        # Save lr 
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

def epoch_sequence_embeddeding_autoencoder(model, loader, config, is_train, loss_function, optimizer = None):
    tot_loss = 0
    for batch in loader:
        if config['train_with_info_data']:
            x = batch[0].to(config['device'])
            x_info = batch[1].to(config['device']).unsqueeze(-1).unsqueeze(-1)
        else:
            x = batch.to(config['device'])
            x_info = None
                        
        if(is_train): # Executed during training
            # Zeros past gradients and forward step
            optimizer.zero_grad()
            
            # Forward pass
            x_r, sequence_embedding, cell_state = model(x, x_info)
            
            # Loss computation. 
            # N.B.The sequence embedding are always passed but not always used. Their used depends by regularize_sequence_embedding parameter in config
            autoencoder_loss = sequence_autoencoder_loss_function(x, x_r, sequence_embedding, loss_function, config)
            
            # Backward and optimization pass
            autoencoder_loss.backward()
            optimizer.step()
        else: # Executed during testing
            with torch.no_grad(): # Deactivate the tracking of the gradient
                x_r, sequence_embedding, cell_state = model(x)
                autoencoder_loss = sequence_autoencoder_loss_function(x, x_r, sequence_embedding, loss_function, config)
        
        # The multiplication serve to compute the average loss over the dataloader
        tot_loss += autoencoder_loss * x.shape[0]
    
    # The division serve to compute the average loss over the dataloader
    tot_loss = tot_loss / len(loader.sampler)
     
    return tot_loss


def sequence_autoencoder_loss_function(original_sequence, reconstructed_sequence, sequence_embedding, loss_function, config):
    if config['compute_loss_spectra_by_spectra']:
        # The MSE is computed spectra by spectra
        # i.e. the first spectra of the original sequence is computed with the first spectra of the reconstructed sequence
        tmp_loss = 0
        for i in range(original_sequence.shape[1]):
            tmp_loss += loss_function(original_sequence[:, i, :], reconstructed_sequence[:, i, :])
    else:
        # MSE between the two entire sequence
        tmp_loss = loss_function(original_sequence, reconstructed_sequence)
    
    # (OPTIONAL) L2 regularization on sequence embedding value
    if config['regularize_sequence_embedding']:
        tmp_loss += torch.pow(torch.sum(sequence_embedding), 2)
    
    return tmp_loss


def update_autoencoder_log_dict(ae_loss_list, log_dict):
    log_dict["SE_AE_loss_train"] = ae_loss_list[0]
    log_dict["SE_AE_loss_validation"] = ae_loss_list[1]
    
    loss_string =  "\tTrain loss     : {}".format(ae_loss_list[0]) + "\n"
    loss_string += "\tValidation loss: {}".format(ae_loss_list[1])
    
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


def save_idx_to_artifact(idx_dict, artifact, run):
    idx_file_path = "TMP_File/idx_dict.pkl"
    a_file = open(idx_file_path, "wb")
    pickle.dump(idx_dict, a_file)
    a_file.close()
    
    artifact.add_file(idx_file_path)