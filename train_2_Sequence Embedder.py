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
    use_spectra_embedder = False,
    query_embedding_size = 256,
    key_embedding_size = 256,
    value_embedding_size = 256,
    use_activation_in_spectra_embedder = True,
    # Multihead attention parameters
    use_attention = False,
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
    use_spectra_embedder = False,
    query_embedding_size = 256,
    key_embedding_size = 256,
    value_embedding_size = 256,
    use_activation_in_spectra_embedder = True,
    # Multihead attention parameters
    use_attention = False,
    num_heads = 4,
    multihead_attention_dropout = 0,
    multihead_attention_bias = True,
    kdim = 256,
    vdim = 256,
    # LSTM Parameters
    sequence_embedding_size = 2,
    LSTM_bias = True,
    LSTM_dropout = 0,
)
    
decoder_config = dict(
        decoder_type = 2,
        sequence_embedding_size = embedder_config['sequence_embedding_size'],
        decoder_LSTM_output_size = embedder_config['sequence_embedding_size'],
        LSTM_bias = False,
        LSTM_dropout = 0,
        proj_size = 0,
        use_reconstruction_layer = True
)

model_config = {'embedder_config':embedder_config, 'decoder_config':decoder_config}

untrained_model = build_and_log_Sequence_Embedder_autoencoder_model(project_name, model_config)

#%% Train model

dataset_config = dict(
    # Artifacts info
    artifact_name = 'jesus_333/Seletech VAE Spectra/Dataset_Spectra_1',
    version = 'latest',
    return_other_sensor_data = True,
    spectra_file_name = '[2021-08-05_to_11-26]All_PlantSpectra.csv',
    water_file_name = '[2021-08-05_to_11-26]PlantTest_Notes.csv',
    ht_file_path = '[2021-08-05_to_11-26]All_PlantHTSensor.csv',
    ht_timestamp_path = 'jesus_ht_timestamp.csv', 
    spectra_timstamp_path = 'jesus_spectra_timestamp.csv',
    n_std = 1,
    binary_label = False,
    # Normalization settings
    normalize_trials = 2,
    normalize_info_avg = True,
    # Sequence construction parameters
    sequence_length = 60,
    shift = 10,
)

"""
v1 = embedding size = 8. USE Spectra embedder 256. USE multihead attention
v2 = embedding size = 8. NO spectra embedder. NO multihead attention
v3 Skinny model. Embedding size = 2. LSTM encoder bias = False.
v4 Projection model (256) WITH activation. Embedding size = 2. REMOVED (collapse in 1 point)
v5 Projection model (256) NO activation. Embedding size = 2. LSTM encoder bias = True. REMOVED (collapse in 2 point)
v6 Skinny model. Embedding size = 2. LSTM encoder bias = True.
v7 Attention model (2 head). Embedding size = 2. LSTM encoder bias = True. REMOVED (alternate between collapse and some linear distribution)
v8 Skinny model. Embedding size = 3. LSTM encoder bias = True.

Decoder type 2:
v11: skinny model. Embedding size = 2. LSTM encoder bias = True. (REMOVED)

Decoder type 3:
v9: skinny model. Embedding size = 8. LSTM encoder bias = True.
v10: skinny model. Embedding size = 2. LSTM encoder bias = True.
"""

training_config = dict(
    model_artifact_name = 'SequenceEmbedder_AE',
    version = 'v10', # REMEMBER ALWAYS TO CHECK THE VERSION
    split_percentage_list = [0.8, 0.05, 0.15], # Percentage of train/test/validation
    # Numerical Hyperparameter
    batch_size = 32,
    lr = 1e-1,
    epochs = 20,
    use_scheduler = True,
    gamma = 0.9, # Parameter of the lr exponential scheduler
    optimizer_weight_decay = 1e-2,
    # Additional option for the training
    train_with_info_data = True,
    compute_loss_spectra_by_spectra = True,
    regularize_sequence_embedding = False,
    # Support stuff (device, log frequency etc)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    log_freq = 5,
    epoch_to_save_model = 1,
    print_var = True,
    dataset_config = dataset_config,
    debug = True,
)

# Train model
model = train_and_log_SE_model(project_name, training_config)


"""
Trained model
v16 (da v6 untrained). train_with_info_data = False, compute_loss_spectra_by_spectra = True, regularize_sequence_embedding = True
Decente

v17 (da v6 untrained). train_with_info_data = False, compute_loss_spectra_by_spectra = False, regularize_sequence_embedding = False
Decente (similie a v16 con ordine di grandezza diverso)

v21 e v22 (da v9 untrained). train_with_info_data = False, compute_loss_spectra_by_spectra = False, regularize_sequence_embedding = False.
v21 ---> sequence = 15. v22 sequence ---> 60

v23 (da v10 untrained). train_with_info_data = False, compute_loss_spectra_by_spectra = False, regularize_sequence_embedding = False. sequence ---> 60
"""

#%%

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from support.datasets import SpectraSequenceDataset
from support.wandb_init_V2 import load_dataset_from_artifact, load_trained_model_from_artifact
from support.wandb_visualization import compute_embedding

dataset_config = dict(
    # Artifacts info
    artifact_name = 'jesus_333/Seletech VAE Spectra/Dataset_Spectra_1',
    version = 'latest',
    return_other_sensor_data = True,
    spectra_file_name = '[2021-08-05_to_11-26]All_PlantSpectra.csv',
    water_file_name = '[2021-08-05_to_11-26]PlantTest_Notes.csv',
    ht_file_path = '[2021-08-05_to_11-26]All_PlantHTSensor.csv',
    ht_timestamp_path = 'jesus_ht_timestamp.csv', 
    spectra_timstamp_path = 'jesus_spectra_timestamp.csv',
    n_std = 1,
    binary_label = False,
    # Normalization settings
    normalize_trials = 1,
    normalize_info_avg = False,
    # Sequence construction parameters
    sequence_length = 15,
    shift = 10,
    # Other parameters
    batch_size = 32
)

data = load_dataset_from_artifact(dataset_config)
dataset = SpectraSequenceDataset(data[0], dataset_config, data[4])
loader = make_dataloader(dataset, dataset_config)

model_config = dict(
    artifact_name = 'jesus_333/Seletech VAE Spectra/SequenceEmbedder_AE_trained',
    version = 'latest',
    epoch_of_model = -1,
    model_file_name = 'model'
)

model, model_config, idx_dict = load_trained_model_from_artifact(model_config)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
embedder = model.embedder.to(device)
# embedder = untrained_model.embedder.to(device)

plot_config = dict(
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
)

embedding = compute_embedding(embedder, loader, plot_config)

if embedding.shape[1] > 2:
    p_tsne = TSNE(n_components = 2, learning_rate='auto', init='random').fit_transform(embedding)      
    p_pca = PCA(n_components=2).fit_transform(embedding)
else:
     p_tsne = embedding
     p_pca = embedding

print(p_tsne.shape)
print(p_pca.shape)

color = []
for batch in loader:
    y = batch[2]
    color.append(y)
color = torch.cat(color)
s = 8

if embedding.shape[1] > 2:
    plt.figure(figsize = (10, 8))
    sc = plt.scatter(p_tsne[:,0], p_tsne[:,1], c = color, s = s,  cmap = 'Greens_r')
    plt.colorbar(sc)

    plt.figure(figsize = (10, 8))
    sc = plt.scatter(p_pca[:,0], p_pca[:,1], c = color, s = s,  cmap = 'Greens_r')
    plt.colorbar(sc)
else:
    plt.figure(figsize = (10, 8))
    sc = plt.scatter(embedding[:,0], embedding[:,1], s = s, c = color,  cmap = 'Greens_r')
    plt.colorbar(sc)

#%%