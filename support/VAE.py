"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)
"""

#%% Imports

import numpy as np

import torch
from torch import nn

from support.embedding import Attention1D

#%%  VAE Single MEMS

class SpectraVAE_Single_Mems(nn.Module):
    
    def __init__(self, N, hidden_space_dimension, print_var = False, use_as_autoencoder = False):
        """
        N = Input length
        hidden_space_dimension = Dimension of the hidden (latent) space. Defaul is 2 
        """
        
        super().__init__()

        self.encoder = SpectraVAE_Encoder_Single_Mems(N, hidden_space_dimension, print_var, use_as_autoencoder)
        
        self.decoder = SpectraVAE_Decoder_Single_Mems(N, hidden_space_dimension, print_var, use_as_autoencoder)
        
        self.hidden_space_dimension = hidden_space_dimension
        
        if(print_var): print("Number of trainable parameters (VAE) = ", sum(p.numel() for p in self.parameters() if p.requires_grad), "\n")
        
    def forward(self, x):
        z_mu, z_log_var = self.encoder(x)

        z = self.reparametrize(z_mu, z_log_var)
        
        x_mean, x_log_var = self.decoder(z)
        
        return x_mean, x_log_var, z_mu, z_log_var
    
    def reparametrize(self, mu, log_var):
      """
      Reparametrization Trick to allow gradients to backpropagate from the stochastic part of the model
      """
        
      sigma = torch.exp(0.5 * log_var)
      noise = torch.randn(size = (mu.size(0), mu.size(1)))
      noise = noise.type_as(mu) # Setting noise to be .cuda when using GPU training 
      
      return mu + sigma * noise


class SpectraVAE_Encoder_Single_Mems(nn.Module):
    
    def __init__(self, N, hidden_space_dimension = 2, print_var = False, use_as_autoencoder = False):
        """
        Encoder of the VAE 
        N = Input length
        hidden_space_dimension = Dimension of the hidden (latent) space. Defaul is 2 
        """
        
        super().__init__()
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(N, 128),
            torch.nn.SELU(),
            torch.nn.Linear(128, 36),
            torch.nn.SELU(),
            torch.nn.Linear(36, hidden_space_dimension * 2),
        )

        
        self.hidden_space_dimension = hidden_space_dimension
        
        if(print_var): print("Number of trainable parameters (VAE - ENCODER) = ", sum(p.numel() for p in self.parameters() if p.requires_grad), "\n")
        
    def forward(self, x):
        x = self.encoder(x)
        mu = x[:, 0:self.hidden_space_dimension]
        log_var = x[:, self.hidden_space_dimension:]
        
        return mu, log_var


class SpectraVAE_Decoder_Single_Mems(nn.Module):
    
    def __init__(self, N, hidden_space_dimension = 2, print_var = False, use_as_autoencoder = False):
        """
        Encoder of the VAE. The output distribution is hypothesized gaussian so the decoder will return two value: mean and distributoin.
        (More info: https://arxiv.org/pdf/2006.13202.pdf)

        Input parameters:
          N = Input length
          hidden_space_dimension = Dimension of the hidden (latent) space. Default is 2 
        """
        
        super().__init__()
        
        self.inner_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_space_dimension, 36),
            torch.nn.SELU(),
            torch.nn.Linear(36, 128),
            torch.nn.SELU()
        )

        self.output_layer_mean = torch.nn.Linear(128, N)
        self.output_layer_log_var = torch.nn.Linear(128, 1)
        
        self.hidden_space_dimension = hidden_space_dimension
        
        if(print_var): print("Number of trainable parameters (VAE - DECODER) = ", sum(p.numel() for p in self.parameters() if p.requires_grad), "\n")
        
    def forward(self, z):
        x = self.inner_layer(z)
        
        x_mean = self.output_layer_mean(x)
        x_log_var = self.output_layer_log_var(x)
        
        return x_mean, x_log_var
    
#%% VAE Double MEMS

class SpectraVAE_Double_Mems(nn.Module):
    
    def __init__(self, N_mems_1, N_mems_2, hidden_space_dimension, use_as_autoencoder = False, use_bias = True, print_var = False,):
        """
        N = Input length
        hidden_space_dimension = Dimension of the hidden (latent) space. Defaul is 2 
        """
        
        super().__init__()

        self.encoder = SpectraVAE_Encoder_Double_Mems(N_mems_1, N_mems_2, hidden_space_dimension, print_var, use_as_autoencoder, use_bias)
        
        self.decoder = SpectraVAE_Decoder_Double_Mems(N_mems_1, N_mems_2, hidden_space_dimension, print_var, use_as_autoencoder, use_bias)
        
        self.hidden_space_dimension = hidden_space_dimension
        
        self.use_as_autoencoder = use_as_autoencoder
        
        self.trainable_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        if(print_var): print("Number of trainable parameters (VAE) = ", sum(p.numel() for p in self.parameters() if p.requires_grad), "\n")
       
        
    def forward(self, x1, x2):
        if self.use_as_autoencoder:
            z = self.encoder(x1, x2)
            x_1_r, x_2_r = self.decoder(z)
            return x_1_r, x_2_r, z
        else:
            z_mu, z_log_var = self.encoder(x1, x2)
    
            z = self.reparametrize(z_mu, z_log_var)
            
            x_mean_1, x_log_var_1, x_mean_2, x_log_var_2 = self.decoder(z)
            
            return x_mean_1, x_mean_2, x_log_var_1, x_log_var_2, z_mu, z_log_var
    
    
    def reparametrize(self, mu, log_var):
      """
      Reparametrization Trick to allow gradients to backpropagate from the stochastic part of the model
      """
        
      sigma = torch.exp(0.5 * log_var)
      noise = torch.randn(size = (mu.size(0), mu.size(1)))
      noise = noise.type_as(mu) # Setting noise to be .cuda when using GPU training 
      
      return mu + sigma * noise


class SpectraVAE_Encoder_Double_Mems(nn.Module):
    
    def __init__(self, N_mems_1, N_mems_2, hidden_space_dimension = 2, print_var = False, use_as_autoencoder = False, use_bias = True):
        """
        Encoder of the VAE 
        N = Input length
        hidden_space_dimension = Dimension of the hidden (latent) space. Defaul is 2 
        """
        
        super().__init__()

        self.input_mems_1 = torch.nn.Sequential(torch.nn.Linear(N_mems_1, 64, bias = use_bias), torch.nn.SELU())
        self.input_mems_2 = torch.nn.Sequential(torch.nn.Linear(N_mems_2, 64, bias = use_bias), torch.nn.SELU())
        
        if(use_as_autoencoder):
            self.inner_layers = torch.nn.Sequential(
                torch.nn.Linear(128, 36, bias = use_bias),
                torch.nn.SELU(),
                torch.nn.Linear(36, hidden_space_dimension, bias = use_bias),
            )
        else:
            self.inner_layers = torch.nn.Sequential(
                torch.nn.Linear(128, 36, bias = use_bias),
                torch.nn.SELU(),
                torch.nn.Linear(36, hidden_space_dimension * 2, bias = use_bias),
            )

        self.N_mems_1 = N_mems_1
        self.N_mems_2 = N_mems_2
        self.hidden_space_dimension = hidden_space_dimension
        self.use_as_autoencoder = use_as_autoencoder
        self.trainable_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        if(print_var): print("Number of trainable parameters (VAE - ENCODER) = ", sum(p.numel() for p in self.parameters() if p.requires_grad), "\n")
        
    def forward(self, x1, x2):
        # N.b. This is the feedforward encoder. The input has dimension "n. batch x signal length"
    
        x1 = self.input_mems_1(x1)
        x2 = self.input_mems_2(x2)
        
        x = torch.cat((x1, x2), 1) 
        z = self.inner_layers(x)
        
        if(self.use_as_autoencoder):
            return z
        else:  
            mu = z[:, 0:self.hidden_space_dimension]
            log_var = z[:, self.hidden_space_dimension:]
        
            return mu, log_var


class SpectraVAE_Decoder_Double_Mems(nn.Module):
    
    def __init__(self, N_mems_1, N_mems_2, hidden_space_dimension = 2, print_var = False, use_as_autoencoder = False, use_bias = True):
        """
        Encoder of the VAE. The output distribution is hypothesized gaussian so the decoder will return two value: mean and distributoin.
        (More info: https://arxiv.org/pdf/2006.13202.pdf)

        Input parameters:
          N = Input length
          hidden_space_dimension = Dimension of the hidden (latent) space. Default is 2 
        """
        
        super().__init__()
        
        self.inner_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_space_dimension, 36, bias = use_bias),
            torch.nn.SELU(),
            torch.nn.Linear(36, 128, bias = use_bias),
            torch.nn.SELU()
        )

        self.output_layer_mean_mems_1 = torch.nn.Linear(64, N_mems_1, bias = use_bias)
        self.output_layer_mean_mems_2 = torch.nn.Linear(64, N_mems_2, bias = use_bias)
        
        if not use_as_autoencoder:
            self.output_layer_log_var_mems_1 = torch.nn.Linear(64, N_mems_1, bias = use_bias)
            self.output_layer_log_var_mems_2 = torch.nn.Linear(64, N_mems_2, bias = use_bias)
            
            
        self.N_mems_1 = N_mems_1
        self.N_mems_2 = N_mems_2
        self.hidden_space_dimension = hidden_space_dimension
        self.use_as_autoencoder = use_as_autoencoder
        self.trainable_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        if(print_var): print("Number of trainable parameters (VAE - DECODER) = ", self.trainable_parameters, "\n")
        
    def forward(self, z):
        x = self.inner_layer(z)
        
        x1 = x[:, 0:int(x.shape[1]/2)]
        x2 = x[:, int(x.shape[1]/2):]
        
        x_mean_1 = self.output_layer_mean_mems_1(x1)
        

        x_mean_2 = self.output_layer_mean_mems_2(x2)
        
        if(self.use_as_autoencoder):
            return x_mean_1, x_mean_2
        else:
            x_log_var_1 = self.output_layer_log_var_mems_1(x1)
            x_log_var_2 = self.output_layer_log_var_mems_2(x2)
            
            return x_mean_1, x_log_var_1, x_mean_2, x_log_var_2

#%% Attention VAE

class AttentionVAE(nn.Module):
    
    def __init__(self, N_mems_1, N_mems_2, hidden_space_dimension, embedding_size = 64, print_var = False, use_as_autoencoder = False):
        """
        Modified versione of the SpectraVAE_Double_Mems where the input are first passed through an attention module
        """
        
        super().__init__()
        
        # Attention module
        self.attention_module = AttentionHead(N_mems_1, N_mems_2, embedding_size)
        
        # VAE
        self.vae = SpectraVAE_Double_Mems(embedding_size * 2, embedding_size * 2, hidden_space_dimension, print_var, use_as_autoencoder)
        
        # Declare again the decoder to have the right output dimension
        self.vae.decoder = SpectraVAE_Decoder_Double_Mems(N_mems_1, N_mems_2, hidden_space_dimension, print_var, use_as_autoencoder)
        
    def forward(self, x1, x2):
        x1, x2 = self.attention_module(x1, x2)
        
        return self.vae(x1, x2)
    
    
class AttentionHead(nn.Module):
    
    def __init__(self,  N_mems_1, N_mems_2, embedding_size = 64):
        super().__init__()
        
        # Self attention (mems1 on mems1)
        self.attention_1_1 = Attention1D(N_mems_1, embedding_size, use_activation = False)
        # Self attention (mems2 on mems2)
        self.attention_2_2 = Attention1D(N_mems_2, embedding_size, use_activation = False)
        # Attention mems1 on mems2
        self.attention_1_2 = Attention1D(N_mems_1, embedding_size, N_mems_2, use_activation = False)
        # Attention mems2 on mems1
        self.attention_2_1 = Attention1D(N_mems_2, embedding_size, N_mems_1, use_activation = False)
        
    def forward(self, x1, x2):
        x11 = self.attention_1_1(x1).squeeze()
        x22 = self.attention_2_2(x2).squeeze()
        x12 = self.attention_1_2(x1, x2).squeeze()
        x21 = self.attention_2_1(x2, x1).squeeze()
        
        x1 = torch.cat((x11, x12), 1)
        x2 = torch.cat((x22, x21), 1)
        
        return x1, x2
    
#%% Other functions

def sampling_latent_space(mu, log_var):
  std = np.sqrt(np.exp(log_var.detach().numpy()))
  points = np.zeros((mu.shape[0], 2))

  for i in range(mu.shape[0]):
    z1 = np.random.normal(loc = mu[i, 0].detach().numpy(), scale = std[i, 0])
    z2 = np.random.normal(loc = mu[i, 1].detach().numpy(), scale = std[i, 1])


    points[i, 0] = z1
    points[i, 1] = z2

  return points