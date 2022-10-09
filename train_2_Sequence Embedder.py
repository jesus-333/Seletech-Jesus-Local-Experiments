# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 18:24:02 2022

@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)
"""

#%%

import sys
sys.path.insert(0, 'support')

import wandb

import torch
import wandb

from support.wandb_init_V1 import make_dataloader
from support.wandb_init_V2 import build_and_log_Sequence_Embedder_model
from support.wandb_init_V2 import build_Sequence_Embedder_model, load_dataset_local, split_dataset
from support.wandb_training_V2 import train_and_log_model
from support.wandb_visualization import bar_loss_wandb_V1

#%% Wandb login and log file inizialization

wandb.login()

project_name = "Seletech VAE Spectra"

#%% Build model

model_config = dict(
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

build_and_log_Sequence_Embedder_model(project_name, model_config)