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
    """
    Embedder of sequence through the multi head attention presented in the paper "Attention is all you need"
    """
    
    def __init__(self, config):
        super().__init__()
        
        if 'dropout' not in config: config['multihead_attention_dropout'] = 0
        
        if 'spectra_length' not in config: spectra_length = 702
        else: spectra_length = config['spectra_length']
        
        self.use_spectra_embedder = config['use_spectra_embedder']
        if config['use_spectra_embedder']: 
            self.query = SpectraEmbedder(702, config['query_embedding_size'], config['use_activation_in_spectra_embedder'])
            self.key = SpectraEmbedder(702, config['key_embedding_size'], config['use_activation_in_spectra_embedder'])
            self.value = SpectraEmbedder(702, config['value_embedding_size'], config['use_activation_in_spectra_embedder'])
            
            input_size = config['query_embedding_size']
            if 'kdim ' not in config: config['kdim'] = config['key_embedding_size']
            if 'vdim ' not in config: config['vdim '] = config['value_embedding_size']
        else: 
            self.spectra_embedder = None
            input_size = 702
            if 'kdim ' not in config: config['kdim'] = spectra_length
            if 'vdim ' not in config: config['vdim '] = spectra_length
        
        self.use_attention = config['use_attention']
        if config['use_attention']:
            self.multi_attention = nn.MultiheadAttention(input_size, config['num_heads'],
                                 dropout = config['multihead_attention_dropout'], bias = config['multihead_attention_bias'], batch_first = True,
                                 kdim = config['kdim'], vdim = config['vdim'], 
                                 )
        else:
            self.multi_attention = None
        
        self.sequence_embedding = nn.LSTM(input_size, config['sequence_embedding_size'], 
                                          batch_first = True, bias = config['LSTM_bias'], dropout = config['LSTM_dropout'])
        
        print("Number of trainable parameters (VAE) = ", sum(p.numel() for p in self.parameters() if p.requires_grad), "\n")
        
    
    def forward(self, x, info_tensor = None):
        """
        x: Input sequence with dimension "B X L X H" where
            B = Batch size
            L = Sequence length
            H = Input size
        info_tensor: (OPTIONAL) tensor with additional information about the input sequence (e.g. water or humidity)
                     Size is "B X L X 1"
        """
        
        if not info_tensor == None:
            x += info_tensor
        
        if self.use_spectra_embedder:
            q = self.query(x)
        else:
            q = x
        
        if self.use_attention:
            if self.use_spectra_embedder:
                k = self.key(x)
                v = self.value(x)
            else:
                k, v = x, x
            
            att_output, att_weights = self.multi_attention(q, k, v)
        else:
            att_output = q
        
        out, (h_n, c_n) = self.sequence_embedding(att_output)
        
        return h_n
    
#%% Clf for training

class Sequence_embedding_clf(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        self.embedder = SequenceEmbedder_V2(config)
        
        self.clf = nn.Linear(config['sequence_embedding_size'], config['n_class'])
        self.log_softmax = nn.LogSoftmax(dim = 1)
        
        print("Number of trainable parameters (VAE) = ", sum(p.numel() for p in self.parameters() if p.requires_grad), "\n")
    
    def forward(self, x):
        x = self.embedder(x).squeeze()
        
        x = self.log_softmax(self.clf(x))
        
        return x
    
#%% Test

if __name__ == "__main__":

    a = torch.rand(4, 5, 30)
    att_1 = nn.MultiheadAttention(30,1, batch_first = True)
    att_2 = nn.MultiheadAttention(30,2, batch_first = True)
    att_5 = nn.MultiheadAttention(30,5, batch_first = True)
    
    b1 = att_1(a,a,a)
    b2 = att_2(a,a,a)
    b5 = att_5(a,a,a)
    
    print("b1.shape: ", b1[0].shape)
    print("b2.shape: ", b2[0].shape)
    print("b5.shape: ", b5[0].shape)
    
    c1 = b1[0]
    c2 = b2[0]
    
    sequence_projection = 2
    rnn = nn.LSTM(30, sequence_projection, batch_first = True)
    
    d1, (d11, d12) = rnn(c1)
    
    tmp_config = dict(
        # Spectra embedder parameters
        use_spectra_embedder = True,
        query_embedding_size = 128,
        key_embedding_size = 128,
        value_embedding_size = 128,
        use_activation_in_spectra_embedder = False,
        # Multihead attention parameters
        use_attention = True,
        num_heads = 1,
        multihead_attention_dropout = 0,
        multihead_attention_bias = True,
        kdim = 128,
        vdim = 128,
        # LST Parameters
        sequence_embedding_size = 2,
        LSTM_bias = False,
        LSTM_dropout = 0
    )
    
    embedder = SequenceEmbedder_V2(tmp_config)
    x = torch.rand((4, 5, 700))
    sequence_embed = embedder(x)
    print("sequence_embed.shape: ", sequence_embed.shape)
    
    tmp_config['n_class'] = 2
    emb_clf = Sequence_embedding_clf(tmp_config)
    
    y = emb_clf(x)