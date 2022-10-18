# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 18:24:02 2022

@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)
"""

#%%

import sys
sys.path.insert(0, 'support')

import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt

from support.wandb_init_V1 import make_dataloader
from support.wandb_init_V2 import build_and_log_Sequence_Embedder_clf_model, build_and_log_Sequence_Embedder_autoencoder_model
from support.wandb_training_Sequence import train_and_log_SE_model

#%% Wandb login and log file inizialization

wandb.login()

project_name = "Seletech VAE Spectra"

#%% Build clf model

model_config = dict(
    # Spectra embedder parameters
    use_spectra_embedder = True,
    query_embedding_size = 256,
    key_embedding_size = 256,
    value_embedding_size = 256,
    use_activation_in_spectra_embedder = True,
    # Multihead attention parameters
    use_attention = True,
    num_heads = 4,
    multihead_attention_dropout = 0,
    multihead_attention_bias = True,
    kdim = 256,
    vdim = 256,
    # LSTM Parameters
    sequence_embedding_size = 8,
    LSTM_bias = False,
    LSTM_dropout = 0,
    # CLf parameters
    n_class = 2,
)

untrained_model = build_and_log_Sequence_Embedder_clf_model(project_name, model_config)

#%% Build auteoncoder model

embedder_config = dict(
    # Spectra embedder parameters
    use_spectra_embedder = True,
    query_embedding_size = 256,
    key_embedding_size = 256,
    value_embedding_size = 256,
    use_activation_in_spectra_embedder = True,
    # Multihead attention parameters
    use_attention = True,
    num_heads = 4,
    multihead_attention_dropout = 0,
    multihead_attention_bias = True,
    kdim = 256,
    vdim = 256,
    # LSTM Parameters
    sequence_embedding_size = 8,
    LSTM_bias = False,
    LSTM_dropout = 0,
)

decoder_config = dict(
    decoder_type = 1,
    sequence_embedding_size = embedder_config['sequence_embedding_size'],
    LSTM_bias = False,
    LSTM_dropout = 0,
    decoder_LSTM_output_size = 256
)

model_config = {'embedder_config':embedder_config, 'decoder_config':decoder_config}

untrained_model = build_and_log_Sequence_Embedder_autoencoder_model(project_name, model_config)

#%%

training_config = dict(
    model_artifact_name = 'SequenceEmbedder_clf',
    version = 'latest', # REMEMBER ALWAYS TO CHECK THE VERSION
    batch_size = 32,
    lr = 1e-2,
    epochs = 5,
    use_scheduler = True,
    gamma = 0.75, # Parameter of the lr exponential scheduler
    optimizer_weight_decay = 1e-3,
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    log_freq = 1,
    print_var = True,
    dataset_config = dataset_config
)

# Create dataloader 
train_loader = make_dataloader(dataset_train, training_config)
validation_loader = make_dataloader(dataset_validation, training_config)
test_loader = make_dataloader(dataset_test, training_config)
loader_list =[train_loader, validation_loader]

# Train model
model = train_and_log_SE_model(project_name, loader_list, training_config)

#%%

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
embedder = model.embedder.to(training_config['device'])
tmp_dataset = dataset_train
point = np.zeros((len(tmp_dataset), model_config['sequence_embedding_size']))
color = []
for i in range(len(tmp_dataset)):
    x, y = tmp_dataset[i]
    emb = embedder(x.to(training_config['device']).unsqueeze(0))
    
    emb = emb.squeeze().detach().cpu()
    
    point[i] = emb
    
    if y == 0: color.append('red')
    if y == 1: color.append('blue')
  

if point.shape[1] > 2:
    # p = TSNE(n_components = 2, learning_rate='auto', init='random').fit_transform(point)      
    p = PCA(n_components=2).fit_transform(point)
else:
    p = point
    
plt.figure(figsize = (15, 10))
plt.scatter(p[:, 0], p[:, 1], c = color)
# plt.xlim([0, 4 * 1e-19])
# plt.ylim([0, - 1.5 * 1e-34])

#%%