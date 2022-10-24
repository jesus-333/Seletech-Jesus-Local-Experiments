"""
Created on Fri Oct  7 11:13:09 2022

@author: Alberto Zancanaro (jesus)
@organization: University of Padua
"""

#%% Imports

import torch
from torch import nn

from support.embedding_spectra import SpectraEmbedder

#%% Sequence Embedder (ENCODER)
    
class SequenceEmbedder(nn.Module):
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
        
        print("Number of trainable parameters (Sequence embedder) = ", sum(p.numel() for p in self.parameters() if p.requires_grad), "\n")
        
    
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
        
        # Reorder dimension so the batch is the first dimension
        h_n = h_n.permute(1, 0, 2)
        
        return out, h_n, c_n
    
#%% Sequence DisEmbedder (Decoder)

class Sequence_Decoder_V1(nn.Module):
    """
    Decoder type 1:
        Decoder where the input is always the same for each step of the reconstructed sequence (i.e. I use always the input sequence embedding as output of the LSTM decoder).
        Image scheme: https://www.mdpi.com/energies/energies-15-01061/article_deploy/html/images/energies-15-01061-g002.png
        N.b. This allow the decoder LSTM to have different size for input and hidden state but force to use the same embedding in each reconstruction step.
    
    Decoder type 2:
        Decoder where the input of the next step is the output of the previous step
        Image scheme: https://www.researchgate.net/publication/347217555/figure/fig4/AS:972284625969153@1608822127533/Seq2seq-LSTM-structure.png
    
    Decoder type 3:
        The input is the sum of the previous step and of the embedding
    
    """
    
    def __init__(self, config):
        super().__init__()
        if 'proj_size' not in config: 
            config['proj_size'] = 0
            print('proj_size not find in config. Set to defualt value of 0')
        
        # LSTM decoder
        self.decoder = nn.LSTM(config['sequence_embedding_size'], config['decoder_LSTM_output_size'], 
                                          batch_first = True, bias = config['LSTM_bias'], dropout = config['LSTM_dropout'],
                                          proj_size  = config['proj_size'])
        self.decoder_LSTM_output_size = config['decoder_LSTM_output_size']
        
        # FF layer to transform the output of LSTM in the original signal
        self.use_reconstruction_layer = config['use_reconstruction_layer']
        self.reconstruction_layer = nn.Linear(config['decoder_LSTM_output_size'], 702)
        
        # Set the type of decoder
        self.decoder_type = config['decoder_type']
        
        # Check parameters
        if config['decoder_type'] < 1 or config['decoder_type'] > 3: raise ValueError('decoder_type inside config must be equal to 1, 2 or 3')
        if config['decoder_type'] == 2 or config['decoder_type'] == 3: 
            if config['sequence_embedding_size'] != config['decoder_LSTM_output_size']: raise ValueError('If decoder_type is 2 or 3 the LSTM output must have the same size of the embedding size')
        
        print("Number of trainable parameters (Sequence decoder) = ", sum(p.numel() for p in self.parameters() if p.requires_grad), "\n")
        
    def forward(self, embed, sequence_length):
        device = embed.device
        
        # Define the input for the first time LSTM cell is used
        # N.b. Even if  batch_first = True the hidden and cell state must be define with the first dimension as sequence length and the second dimension as batch size (for batched input)
        h = torch.zeros((1, embed.shape[0], self.decoder_LSTM_output_size)).to(device)
        c = torch.zeros((1, embed.shape[0], self.decoder_LSTM_output_size)).to(device)
        
        # Create the output matrix
        sequence_decoded = torch.zeros((embed.shape[0], sequence_length, self.decoder_LSTM_output_size)).to(device)
        sequence_reconstructed = torch.zeros((embed.shape[0], sequence_length, 702)).to(device)
        
        # Set the first input
        input_encoder = embed
        
        for i in range(sequence_length):
            out, (h, c) = self.decoder(input_encoder, (h, c))
             
            if self.use_reconstruction_layer:
                sequence_decoded[:, i, :] = out.squeeze()
                sequence_reconstructed[:, i, :] = self.reconstruction_layer(out).squeeze()
            else:
                sequence_decoded[:, i, :] = out.squeeze()
                sequence_reconstructed[:, i, :] = out.squeeze()
            
            # If I use the second type of decoder set the input as the previous output
            if self.decoder_type == 2: input_encoder = out
            if self.decoder_type == 3: input_encoder = out + embed
            
        return sequence_reconstructed, sequence_decoded
    
    
class Sequence_Decoder_V2(nn.Module):
    """
    Decoder where the input of the next step is the output of the previous step
    Image scheme: https://www.researchgate.net/publication/347217555/figure/fig4/AS:972284625969153@1608822127533/Seq2seq-LSTM-structure.png
    """
    def __init__(self, config):
        super().__init__()
    
#%% Sequence autoencoder

class SequenceEmbedderAutoencoder(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        self.embedder = SequenceEmbedder(config['embedder_config'])
        
        self.decoder = Sequence_Decoder_V1(config['decoder_config'])
        
        print("Number of trainable parameters (sequence autoencoder) = ", sum(p.numel() for p in self.parameters() if p.requires_grad), "\n")
    
    def forward(self, original_sequence, info_tensor = None):
        """
        original_sequence: Input sequence with dimension "B X L X H" where
            B = Batch size
            L = Sequence length
            H = Input size
        """
        
        # Get the length of the original sequence
        sequence_length = original_sequence.shape[1]
        
        # Compute the embedding (h) of the original sequence
        out, h, c = self.embedder(original_sequence, info_tensor)
        
        # Reconstruct the original sequence from the encoding
        sequence_reconstructed, sequence_decoded = self.decoder(h, sequence_length)
        
        return sequence_reconstructed, h, c
    
#%% Sequence Clf for training

class Sequence_embedding_clf(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        self.embedder = SequenceEmbedder(config)
        
        self.clf = nn.Linear(config['sequence_embedding_size'], config['n_class'])
        self.log_softmax = nn.LogSoftmax(dim = 1)
        
        print("Number of trainable parameters (VAE) = ", sum(p.numel() for p in self.parameters() if p.requires_grad), "\n")
    
    def forward(self, x):
        out, h, c = self.embedder(x)
        
        x = self.log_softmax(self.clf(h.squeeze()))
        
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
        LSTM_dropout = 0,
    )
    
    embedder = SequenceEmbedder(tmp_config)
    x = torch.rand((4, 5, 702))
    sequence_embed = embedder(x)
    out, h, c = sequence_embed
    
    print("x.shape  : ", x.shape)
    print("out.shape: ", out.shape)
    print("h.shape  : ", h.shape)
    print("c.shape  : ", c.shape, "\n")
    
    tmp_config['n_class'] = 2
    emb_clf = Sequence_embedding_clf(tmp_config)
    y = emb_clf(x)
    
    decoder_config = dict(
        decoder_type = 3,
        sequence_embedding_size = tmp_config['sequence_embedding_size'],
        decoder_LSTM_output_size = tmp_config['sequence_embedding_size'],
        LSTM_bias = False,
        LSTM_dropout = 0,
        proj_size = 0,
        use_reconstruction_layer = True
    )

    model_config = {'embedder_config':tmp_config, 'decoder_config':decoder_config}
    emb_ae = SequenceEmbedderAutoencoder(model_config)
    
    print("x (INPUT):\t\t", x.shape)
    
    out = emb_ae(x)