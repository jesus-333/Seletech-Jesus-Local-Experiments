"""
Created on Wed Sep 14 11:24:11 2022

@author: Alberto Zancanaro (jesus)
@organization: University of Padua
"""

#%% Import 

import wandb
import torch

from support.training import advanceEpochV2, advanceEpochV3

#%% Training cycle 

def train_model_wandb(model, optimizer, loader_train, loader_validation, loader_excluded_class, config, lr_scheduler = None):
    if 'log_freq' not in config: config['log_freq'] = 5
    if 'device' not in config: config['device'] = 'cpu'
    
    if 'alpha' not in config: config['alpha'] = 1
    if 'beta' not in config: config['beta'] = 1
    
    wandb.watch(model, log = "all", log_freq = config.log_freq)
    
    
    for epoch in range(config.epochs):
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # Training phase
        training_loss = advance_epoch_envelope(model, optimizer, loader_train, config, 
                               epoch = epoch, is_train = True)
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # Validation phase
        validation_loss = advance_epoch_envelope(model, optimizer, loader_validation, config, 
                               epoch = epoch, is_train = False)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # Excluded class (i.e. the type of spectra not used for the training)
        excluded_loss = advance_epoch_envelope(model, optimizer, loader_excluded_class, config, 
                               epoch = epoch,  is_train = False)
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # Divide loss and wandb log
        
        loss_string = ""
        log_dict = {'learning_rate': optimizer.param_groups[0]['lr']}
        if config.use_as_autoencoder: # Loss in case I train an autoencoder
            log_dict["reconstruction_loss_train"] = training_loss
            log_dict["reconstruction_loss_validation"] = validation_loss
            log_dict["reconstruction_loss_excluded"] = excluded_loss
            loss_string += "\treconstruction_loss_train:\t\t" + str(training_loss.detach().numpy()) + "\n"
            loss_string += "\treconstruction_loss_validation:\t" + str(validation_loss.detach().numpy()) + "\n"
            loss_string += "\treconstruction_loss_excluded:\t\t" + str(excluded_loss.detach().numpy())
        else: # Loss in case I train a VAE
            log_dict, loss_string = divide_VAE_loss(training_loss, "_training", log_dict, loss_string)
            log_dict, loss_string = divide_VAE_loss(validation_loss, "_validation", log_dict, loss_string)
            log_dict, loss_string = divide_VAE_loss(excluded_loss, "_excluded", log_dict, loss_string)
        
        # Log data on wandb
        wandb.log(log_dict)
        
        # Update learning rate (if a scheduler is provided)
        if lr_scheduler is not None: lr_scheduler.step()
        
        if config.print_var: 
            print("Epoch: {}".format(epoch))
            print(loss_string)


def advance_epoch_envelope(model, optimizer, loader, config, epoch, is_train):
    if config.use_as_autoencoder: # Train/tested as autoencoder
        tmp_loss = advanceEpochV3(model, config.device, loader, optimizer, 
                                  is_train = is_train, double_mems = True)
        
    else: # Train/tested as VAE
        tmp_loss = advanceEpochV2(model, config.device, loader, optimizer, 
                                  is_train = is_train, alpha = config.alpha, beta = config.beta)
        
    return tmp_loss
        

def divide_VAE_loss(vae_loss_list, label, log_dict, loss_string):
    """
    Since the operation to divide the VAE loss is always the same for the 3 dataset I write a function to avoid a lot of repeated code
    """
    
    # Temporary dict to store the actual loss
    tmp_dict = {}
    tmp_dict = {"total_loss" + label: vae_loss_list[0],
                "reconstruction_loss" + label: vae_loss_list[1],
                "kl_loss" + label: vae_loss_list[2]}
    
    # Merge the dictionary with the old loss with the dictionary with the new loss
    log_dict = {**log_dict, **tmp_dict}
    
    # Update loss string
    loss_string += "\ttotal_loss" + label + ":\t" + str(vae_loss_list[0].detach().numpy()) + "\n"
    loss_string += "\t\treconstruction_loss" + label + ":\t" + str(vae_loss_list[1].detach().numpy()) + "\n"
    loss_string += "\t\tkl_loss" + label + ":\t" + str(vae_loss_list[2].detach().numpy())
    
    return log_dict, loss_string

#%% Other functions

def save_model_onnx(model, model_path = "model.onnx"):
    model.eval()
    torch.onnx.export(model, None, model_path)
    wandb.save(model_path)
    
def save_model_pytorch(model, model_path):
    model.eval()
    
    torch.save(model.state_dict(), model_path)
    wandb.save(model_path)