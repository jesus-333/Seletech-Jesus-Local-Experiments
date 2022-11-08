"""
Created on Tue Nov  8 16:00:42 2022

@author: Alberto Zancanaro (jesus)
@organization: University of Padua

Train script for NLP spectra embedder
"""

#%% Imports

import sys
sys.path.insert(0, 'support')

import numpy as np
import wandb

from support.wandb_init_V2 import build_and_log_spectra_embedder_NLP

#%% Wandb login and log file inizialization

wandb.login()

project_name = "Seletech VAE Spectra"

#%% Build model

model_config = dict(
    input_size = 700,
    embedding_size = 2,
    type_embedder= 'skipGram',
    window_size = 2,
    debug = True
    )

build_and_log_spectra_embedder_NLP(project_name, model_config)