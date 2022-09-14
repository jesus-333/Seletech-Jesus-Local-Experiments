"""
Created on Wed Sep 14 11:24:11 2022

@author: Alberto Zancanaro (jesus)
@organization: University of Padua
"""

#%% Import 

import wandb

from support.training import advanceEpochV2, advanceEpochV3

#%% Training cycle 

def train_model_wandb(model, optimizer, loader_train, loader_validation, loader_excluded_class, config):
    if 'log_freq' not in config: config['log_freq'] = 5
    if 'config.device' not in config: config['config.device'] = 'cpu'
    
    if 'alpha' not in config: config['alpha'] = 1
    if 'beta' not in config: config['beta'] = 1
    
    wandb.watch(model, log = "all", log_freq = config.log_freq)
    
    
    for epoch in range(config.epochs):
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # Training phase
        advance_epoch_envelope(model, optimizer, loader_train, config, epoch)
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # Validation phase
        advance_epoch_envelope(model, optimizer, loader_validation, config, epoch)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # Excluded class (i.e. the type of spectra not used for the training)
        advance_epoch_envelope(model, optimizer, loader_excluded_class, config, epoch)


def advance_epoch_envelope(model, optimizer, loader, config, epoch, label = ""):
    if config.use_as_autoencoder: # Train/tested as autoencoder
        # Advance the epoch
        tmp_loss_recon = advanceEpochV3(model, config.device, 
                                             loader, optimizer, 
                                             is_train = True, double_mems = True)
        # Log the data on wandb
        log_dict = {"epoch": epoch, "reconstruction_loss" + label: tmp_loss_recon}
        wandb.log(log_dict)
    
    else: # Train/tested as VAE
        # Advance the epoch
        loss_list = advanceEpochV2(model, config.device, 
                                   loader, optimizer, 
                                   is_train = True, alpha = config.alpha, beta = config.beta)
        
        # Log the data on wandb
        log_dict = {"epoch": epoch, 
                    "total_loss" + label: loss_list[0],
                    "reconstruction_loss" + label: loss_list[1],
                    "kl_loss" + label: loss_list[2]}
        wandb.log(log_dict)