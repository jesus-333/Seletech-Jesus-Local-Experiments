"""
Created on Mon Sep 12 11:51:56 2022

@author: jesus
"""

#%% Import

import numpy as np
import torch
from torch import nn

#%% Simple feedforward

class SpectraEmbedder(nn.Module):
    
    def __init__(self, input_size, embedding_size = 2, use_activation = False):
        """
        Simple feedforward network that project the input in another space.
        Eventually a non linear operation could be activated.
        """
        super().__init__()
        
        self.embedder = nn.Linear(input_size, embedding_size)
        self.activation = nn.SELU()
        
        self.use_activation = use_activation
        
    def forward(self, x):
        x = self.embedder(x)
        
        if self.use_activation:
            return self.activation(x)
        else:
            return x

    
class Attention1D(nn.Module): 
    def __init__(self, input_size, embedding_size = 2, input_size_2 = None, use_activation = False):
        """
        Modified version of self attention explained in https://arxiv.org/pdf/1805.08318.pdf
        This version is used to work with 1D array, NOT SEQUENCE OF DATA. 
        
        It can perform both self attention or attention between 2 vectors.
        If input_size_2 it isn't specified OR the input x2 is notused it will perform self attention on x1. 
        To perform attention between x1 and x2 both input_size_2 and x2 must be different from None
        
        """
        super().__init__()
        
        self.softmax = nn.Softmax(dim = 2)
        
        self.input_size_2 = input_size_2
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Embedder for qury/key/value
        
        self.query_embedder = SpectraEmbedder(input_size, embedding_size, use_activation)
        if input_size_2 == None: self.key_embedder = SpectraEmbedder(input_size, embedding_size, use_activation)
        else: self.key_embedder = SpectraEmbedder(input_size_2, embedding_size, use_activation)
        self.value_embedder = SpectraEmbedder(input_size, embedding_size, use_activation)
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        
    def forward(self, x1, x2 = None):
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Compute the query, key and value embedding
        # The name f,g and h came from the paper
        
        # Query
        f = self.query_embedder(x1)
        
        # Key
        if x2 == None or self.input_size_2 == None: g = self.key_embedder(x1)
        else: g = self.key_embedder(x2)
        
        # Value
        h = self.value_embedder(x1)
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Create the attention map
        s = torch.bmm(f.unsqueeze(2), g.unsqueeze(1))
        s = self.softmax(s)
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Compute the output
        o = torch.bmm(s, h.unsqueeze(2))
        
        return o

#%% NLP Embedding

class skipGram(nn.Module):
    
    def __init__(self, config):
        """
        Class that implement the skipgram embedding used in NLP application.
        It is applied to spectra.
        """
        
        super().__init__()
        
        # Embedder layer (input)
        self.embedder = nn.Linear(config['input_size'], config['embedding_size'])
        
        # Reconstruction layers (output)
        self.output_list = nn.ModuleList()
        for i in range(config['window_size']):
            self.output_layer_list.append(nn.Linear(config['embedding_size'], config['input_size']))
        
    
    def forward(self, x):
        embed = self.embedder(x)
        
        output_list = []
        for output_layer in self.output_layer_list:
            tmp_output = output_layer(embed)
            output_list.append(tmp_output)
        
        return output_list
    
    def embed(self, x):
        return self.embedder(x)