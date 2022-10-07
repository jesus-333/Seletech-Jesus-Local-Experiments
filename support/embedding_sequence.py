"""
Created on Fri Oct  7 11:13:09 2022

@author: Alberto Zancanaro (jesus)
@organization: University of Padua
"""

#%% Imports

import torch
from torch import nn

from support.embedding_spectra import SpectraEmbedder

#%%

class SequenceEmbedder(nn.Module):
    """
    Project sequence of spectra inside a smaller space
    """
    def __init__(self, config):
        super().__init__()
        
        if config['use_spectra_embedder']: 
            self.spectra_embedder = SpectraEmbedder(700, config['spectra_embedding_size'], config['use_activation_in_spectra_embedder'])
            input_size = config['spectra_embedding_size']
        else: 
            self.spectra_embedder = None
            input_size = 700
        
        self.sequence_embedding = nn.LSTM(input_size, config['sequence_embedding_size'], batch_first = True)

        
    def forward(self, x):
        return x
    
    
class SequenceEmbedder_V2(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        self.multi_attention = nn.MultiheadAttention(config['embed_dim'], config['num_heads'],
                                                     bias = config['use_bias'], batch_first = True)
        
    
    def forward(self, x):
        return x