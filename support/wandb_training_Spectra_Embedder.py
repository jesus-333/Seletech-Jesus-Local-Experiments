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

#%% Principal function

def train_and_log_SE_model(project_name, config):
    # Get the name for the actual run
    if "skipGram" in config['model_artifact_name']:  run_name = get_run_name('train-SE-skipgram-embedding')
    elif "CBOW" in config['model_artifact_name']:  run_name = get_run_name('train-SE-CBOW-embedding')
    else: raise ValueError("Problem with the type of model you want to build")

    
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
        add_model_to_artifact(model, model_artifact, "TMP_File/model_END.pth")
        run.log_artifact(model_artifact)
        
    return model
