"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)
"""

#%% Imports

import numpy as np

import torch
from torch import nn

    
#%% VAE Double MEMS

class SpectraVAE_Double_Mems_Conv(nn.Module):
    
    def __init__(self, N_mems_1, N_mems_2, hidden_space_dimension, print_var = False):
        """
        N = Input length
        hidden_space_dimension = Dimension of the hidden (latent) space. Defaul is 2 
        """
        
        super().__init__()

        self.encoder = SpectraVAE_Encoder_Double_Mems_Conv(N_mems_1, N_mems_2, hidden_space_dimension, print_var)
        
        self.decoder = SpectraVAE_Decoder_Double_Mems_Conv(self.encoder.mems_1_output_shape, self.encoder.mems_2_output_shape, hidden_space_dimension, print_var)
        
        self.hidden_space_dimension = hidden_space_dimension
        
        if(print_var): print("Number of trainable parameters (VAE) = ", sum(p.numel() for p in self.parameters() if p.requires_grad), "\n")
        
    def forward(self, x1, x2):
        z_mu, z_log_var = self.encoder(x1, x2)

        z = self.reparametrize(z_mu, z_log_var)
        
        x_mean_1, x_log_var_1, x_mean_2, x_log_var_2 = self.decoder(z)
        
        return x_mean_1, x_log_var_1, x_mean_2, x_log_var_2, z_mu, z_log_var
    
    def reparametrize(self, mu, log_var):
      """
      Reparametrization Trick to allow gradients to backpropagate from the stochastic part of the model
      """
        
      sigma = torch.exp(0.5 * log_var)
      noise = torch.randn(size = (mu.size(0), mu.size(1)))
      noise = noise.type_as(mu) # Setting noise to be .cuda when using GPU training 
      
      return mu + sigma * noise


class SpectraVAE_Encoder_Double_Mems_Conv(nn.Module):
    
    def __init__(self, N_mems_1, N_mems_2, hidden_space_dimension = 2, print_var = False):
        """
        Encoder of the VAE 
        N = Input length
        hidden_space_dimension = Dimension of the hidden (latent) space. Defaul is 2 
        """
        
        super().__init__()
        
        # Conv layers for mems1
        self.input_mems_1 = torch.nn.Sequential(
            torch.nn.Conv1d(1, 4, kernel_size = 30, stride = 3),
            torch.nn.BatchNorm1d(4), torch.nn.SELU(),
            torch.nn.Conv1d(4, 8, kernel_size = 15, stride = 3),
            torch.nn.BatchNorm1d(8), torch.nn.SELU(),
            torch.nn.Conv1d(8, 16, kernel_size = 5, stride = 2),
            torch.nn.BatchNorm1d(16), torch.nn.SELU()
        )
        
        # Conv layers for mems2
        self.input_mems_2 = torch.nn.Sequential(
            torch.nn.Conv1d(1, 4, kernel_size = 30, stride = 3),
            torch.nn.BatchNorm1d(4), torch.nn.SELU(),
            torch.nn.Conv1d(4, 8, kernel_size = 15, stride = 3),
            torch.nn.BatchNorm1d(8), torch.nn.SELU(),
            torch.nn.Conv1d(8, 16, kernel_size = 5, stride = 2),
            torch.nn.BatchNorm1d(16), torch.nn.SELU()
        )
        
        # Computete flatten layer dimension and save output shape
        tmp_mems_1_output = self.input_mems_1(torch.ones(1, 1, N_mems_1))
        tmp_mems_2_output = self.input_mems_2(torch.ones(1, 1, N_mems_2))
        flatten_mems_1 = tmp_mems_1_output.shape[1] * tmp_mems_1_output.shape[2]
        flatten_mems_2 = tmp_mems_2_output.shape[1] * tmp_mems_2_output.shape[2]
        self.mems_1_output_shape = tmp_mems_1_output.shape
        self.mems_2_output_shape = tmp_mems_2_output.shape
        
        self.inner_layers = torch.nn.Sequential(
            torch.nn.Linear(flatten_mems_1 + flatten_mems_2, 128),
            torch.nn.SELU(), nn.Dropout(0.25),
            torch.nn.Linear(128, 36),
            torch.nn.SELU(), nn.Dropout(0.25),
            torch.nn.Linear(36, hidden_space_dimension * 2),
        )

        self.N_mems_1 = N_mems_1
        self.N_mems_2 = N_mems_2
        self.hidden_space_dimension = hidden_space_dimension
        
        if(print_var): print("\nNumber of trainable parameters (VAE - ENCODER) = ", sum(p.numel() for p in self.parameters() if p.requires_grad), "\n")
        
    def forward(self, x1, x2):
      x1 = self.input_mems_1(x1)
      x2 = self.input_mems_2(x2)
      x1 = x1.view([x1.shape[0], -1])
      x2 = x2.view([x2.shape[0], -1])
      x = torch.cat((x1, x2), 1)
      x = self.inner_layers(x)
      mu = x[:, 0:self.hidden_space_dimension]
      log_var = x[:, self.hidden_space_dimension:]
      
      return mu, log_var


class SpectraVAE_Decoder_Double_Mems_Conv(nn.Module):
    
    def __init__(self, mems_1_output_shape, mems_2_output_shape, hidden_space_dimension = 2, print_var = False):
        """
        Encoder of the VAE. The output distribution is hypothesized gaussian so the decoder will return two value: mean and distributoin.
        (More info: https://arxiv.org/pdf/2006.13202.pdf)

        Input parameters:
          N = Input length
          hidden_space_dimension = Dimension of the hidden (latent) space. Default is 2 
        """
        
        super().__init__()
        

        self.mems_1_output_shape = mems_1_output_shape
        self.mems_2_output_shape = mems_2_output_shape
        flatten_mems_1 = mems_1_output_shape[1] * mems_1_output_shape[2]
        flatten_mems_2 = mems_2_output_shape[1] * mems_2_output_shape[2]
        
        self.inner_layers = torch.nn.Sequential(
            torch.nn.Linear(hidden_space_dimension, 36),
            torch.nn.SELU(), nn.Dropout(0.25),
            torch.nn.Linear(36, 128),
            torch.nn.SELU(), nn.Dropout(0.25),
            torch.nn.Linear(128, flatten_mems_1 + flatten_mems_2),
            torch.nn.SELU(), nn.Dropout(0.25),
        )
        
        # Conv layers for mems1
        self.output_layer_mems_1 = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(16, 8, kernel_size = 5, stride = 2),
            torch.nn.BatchNorm1d(8), torch.nn.SELU(),
            torch.nn.ConvTranspose1d(8, 4, kernel_size = 15, stride = 3),
            torch.nn.BatchNorm1d(4), torch.nn.SELU(),
        )
        self.mean_mems_1 = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(4, 1, kernel_size = 30, stride = 3), 
            nn.Upsample(size = 300)
        )
        self.variance_mems_1 = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(4, 1, kernel_size = 30, stride = 3), 
            nn.Upsample(size = 300)
        )
        
        # Conv layers for mems2
        self.output_layer_mems_2 = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(16, 8, kernel_size = 5, stride = 2),
            torch.nn.BatchNorm1d(8), torch.nn.SELU(),
            torch.nn.ConvTranspose1d(8, 4, kernel_size = 15, stride = 3),
            torch.nn.BatchNorm1d(4), torch.nn.SELU(),
        )
        self.mean_mems_2 = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(4, 1, kernel_size = 30, stride = 3), 
            nn.Upsample(size = 400)
        )
        self.variance_mems_2 = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(4, 1, kernel_size = 30, stride = 3), 
            nn.Upsample(size = 400)
        )
        
        # self.output_layer_log_var_mems_1 = torch.nn.Linear(64, 1)

        # self.output_layer_mean_mems_2 = torch.nn.Linear(64, N_mems_2)
        # self.output_layer_log_var_mems_2 = torch.nn.Linear(64, 1)
        

        self.hidden_space_dimension = hidden_space_dimension
        
        if(print_var): print("\nNumber of trainable parameters (VAE - DECODER) = ", sum(p.numel() for p in self.parameters() if p.requires_grad), "\n")
        
    def forward(self, z):
        x = self.inner_layers(z)
        
        x1 = x[:, 0:self.mems_1_output_shape[1] * self.mems_1_output_shape[2]]
        x2 = x[:, self.mems_1_output_shape[1] * self.mems_1_output_shape[2]:]
        
        x1 = x1.view([x1.shape[0], self.mems_1_output_shape[1], self.mems_1_output_shape[2]])
        x2 = x2.view([x2.shape[0], self.mems_2_output_shape[1], self.mems_2_output_shape[2]])
        
        x1 = self.output_layer_mems_1(x1)
        x_mean_1 = self.mean_mems_1(x1)
        x_log_var_1 = self.variance_mems_1(x1)

        x2 = self.output_layer_mems_2(x2)
        x_mean_2 = self.mean_mems_2(x2)
        x_log_var_2 = self.variance_mems_2(x2)
        
        return x_mean_1, x_log_var_1, x_mean_2, x_log_var_2
    
    
#%%

def sampling_latent_space(mu, log_var):
    std = np.sqrt(np.exp(log_var.detach().numpy()))
    points = np.zeros((mu.shape[0], 2))
    
    for i in range(mu.shape[0]):
      z1 = np.random.normal(loc = mu[i, 0].detach().numpy(), scale = std[i, 0])
      z2 = np.random.normal(loc = mu[i, 1].detach().numpy(), scale = std[i, 1])
    
    
      points[i, 0] = z1
      points[i, 1] = z2
    
    return points

#%% Test of network architecture

if __name__ == "__main__":
    hidden_space_dimension = 2 
    print_var = False
    
    length_mems_1 = 300
    length_mems_2 = 400
    
    tmp_x1 = torch.rand(3, 1, length_mems_1)
    tmp_x2 = torch.rand(3, 1, length_mems_2)
    
    # - - - - - - - - - 
    # Encoder size during various layer
    
    encoder = SpectraVAE_Encoder_Double_Mems_Conv(length_mems_1, length_mems_2, hidden_space_dimension, print_var)
    mems_1_output_shape = encoder.mems_1_output_shape
    mems_2_output_shape = encoder.mems_2_output_shape
    flatten_1 = mems_1_output_shape[1] * mems_1_output_shape[2]
    flatten_2 = mems_2_output_shape[1] * mems_2_output_shape[2]
    
    x = tmp_x1
    print("ENCODER Input mems 1")
    print(x.shape)
    for layer in encoder.input_mems_1:
        if(isinstance(layer, nn.Conv1d)): 
            x = layer(x)
            print(x.shape)
            
    
    x = tmp_x2
    print("\nENCODER Input mems 2")
    print(x.shape)
    for layer in encoder.input_mems_2:
        if(isinstance(layer, nn.Conv1d)): 
            x = layer(x)
            print(x.shape)
            
    z_mu, z_log_var = encoder(tmp_x1, tmp_x2)     
    # - - - - - - - - - 
    # DECODER size during various layer
    
    decoder = SpectraVAE_Decoder_Double_Mems_Conv(mems_1_output_shape, mems_2_output_shape, hidden_space_dimension, print_var)
    
    tmp_x = torch.rand(3, 448)
    tmp_x1 = tmp_x[:, 0:flatten_1]
    tmp_x2 = tmp_x[:, flatten_1:]
    
    tmp_x1 = tmp_x1.view([tmp_x1.shape[0], mems_1_output_shape[1], mems_1_output_shape[2]])
    tmp_x2 = tmp_x2.view([tmp_x2.shape[0], mems_2_output_shape[1], mems_2_output_shape[2]])
    
    x = tmp_x1
    # print("\nDECODER Input mems 1")
    # print(x.shape)
    # for layer in decoder.output_layer_mean_mems_1:
    #     if(isinstance(layer, nn.ConvTranspose1d)): 
    #         x = layer(x)
    #         print(x.shape)
    
    x_mean_1, x_log_var_1, x_mean_2, x_log_var_2 = decoder(z_mu)
    print(x_mean_1.shape)