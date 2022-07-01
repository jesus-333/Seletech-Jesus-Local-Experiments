"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)
"""

#%% Imports

import numpy as np

import torch
from torch import nn

from support.VAE import SpectraVAE_Double_Mems

#%% Classifier

class SpectraClassifier(nn.Module):
    
    def __init__(self, n_input_neurons, n_outputs):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(n_input_neurons, 64),
            nn.SELU(),
            nn.Linear(64, n_outputs),
            nn.LogSoftmax(dim = 1)
        )
        

    def forward(self, x): 
        return self.classifier(x)
    
#%% Classifier + VAE

class SpectraFramework_FC(nn.Module):
    
    def __init__(self, N_mems_1, N_mems_2, hidden_space_dimension, n_outputs, print_var = False, use_as_autoencoder = False):
        super().__init__()
        
        self.vae = SpectraVAE_Double_Mems(N_mems_1, N_mems_2, hidden_space_dimension, print_var = False, use_as_autoencoder = use_as_autoencoder )
        
        if(use_as_autoencoder): self.clf = SpectraClassifier(hidden_space_dimension, n_outputs)
        else: self.clf = SpectraClassifier(hidden_space_dimension * 2, n_outputs)
        
        self.use_as_autoencoder = use_as_autoencoder
        
        self.trainable_parameters_VAE = sum(p.numel() for p in self.vae.parameters() if p.requires_grad)
        self.trainable_parameters_clf = sum(p.numel() for p in self.clf.parameters() if p.requires_grad)
        
        if(print_var):
            print("Number of trainable parameters (VAE) = ", self.trainable_parameters_VAE)
            print("Number of trainable parameters (clf) = ", self.trainable_parameters_clf, "\n")
    
    def forward(self, x1, x2):
        if(self.use_as_autoencoder):
            x_r_1, x_r_2, z = self.vae(x1, x2)
            
            label = self.clf(z)
            
            return x_r_1, x_r_2, z, label
        else:
            x_mean_1, x_log_var_1, x_mean_2, x_log_var_2, z_mu, z_log_var = self.vae(x1, x2)
            
            z = torch.cat((z_mu, z_log_var), 1)
            
            label = self.clf(z)
            
            return x_mean_1, x_mean_2, x_log_var_1, x_log_var_2, z_mu, z_log_var, label

#%% Test framework

if __name__ == "__main__":
    N_mems_1 = 300
    N_mems_2 = 400
    
    hidden_space_dimension = 8
    n_outputs = 4
    
    x1 = torch.rand((3, N_mems_1))
    x2 = torch.rand((3, N_mems_2))
    
    spectra_framework_autoencoder = SpectraFramework_FC(N_mems_1, N_mems_2, hidden_space_dimension, n_outputs, print_var = True, use_as_autoencoder = True)
    x_r_1, x_r_2, z, label = spectra_framework_autoencoder(x1, x2)
    
    
    spectra_framework_VAE = SpectraFramework_FC(N_mems_1, N_mems_2, hidden_space_dimension, n_outputs, print_var = True, use_as_autoencoder = False)
    x_mean_1, x_log_var_1, x_mean_2, x_log_var_2, z_mu, z_log_var, label = spectra_framework_VAE(x1, x2)
    
    
    