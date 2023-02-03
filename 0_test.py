#%% Imports 

import sys
sys.path.insert(0, 'support')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
import time
import wandb
import torch

from support.datasets import load_spectra_data, load_water_data, create_extended_water_vector
from support.timestamp_function import convert_timestamps_in_dataframe
from support.preprocess import aggregate_HT_data_V1, aggregate_HT_data_V2, choose_spectra_based_on_water_V1
from support.wandb_init_V2 import load_trained_model_from_artifact, load_untrained_model_from_artifact
from support.wandb_init_V2 import load_dataset_from_artifact

#%% Load data

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
    # Normalization settings
    normalize_trials = 2,
    # Dataset config
    window_size = 2,
    batch_size = 32
)

data = load_dataset_from_artifact(dataset_config)

spectra = torch.from_numpy(data[0])
humidity = data[4]
temperature = data[5]

#%%

model_config = dict(
    artifact_name = 'jesus_333/Seletech VAE Spectra/SpectraEmbedder_skipGram_ns_trained',
    version = 'v2',
    epoch_of_model = 10,
    model_file_name = 'model'
)

model, model_config = load_trained_model_from_artifact(model_config)

"""
Trained skipGram_ns
v0: embedding = 2     windos_size = 3
v1: embedding = 16    windos_size = 3
v2: embedding = 16    windos_size = 6
v3: embedding = 2     windos_size = 6
v4: embedding = 32    windos_size = 3
v5: embedding = 32    windos_size = 6
"""

#%%

from support.wandb_visualization import plot_spectra_embedding, fast_compute_embedding, reduce_dimension
from support.wandb_visualization import plot_evolution

plot_config = dict(
    idx_list = np.arange(25) + 1,
    path_weight = 'artifacts/SpectraEmbedder_skipGram_ns_trained-v2/model_',
    dimensionaly_reduction_method = 'pca',
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    figsize = (15, 10),
    color = temperature,
    s = 2,
    cmap = 'Greens_r',
    xlim = [-0.1, 0.1],
    ylim  = [0.1, -0.1]
)

# plot_evolution(spectra, model, plot_config)

model.cuda()
embedding = model(spectra.float().cuda()).detach().cpu().numpy()

if embedding.shape[1] > 2: 
    embedding_reduced = reduce_dimension(embedding, 2, 'pca')
    plot_spectra_embedding(embedding_reduced, plot_config)

    # embedding_reduced = reduce_dimension(embedding, 2, 'tsne')
    # plot_spectra_embedding(embedding, plot_config)
else:
    plot_spectra_embedding(embedding, plot_config)

#%% Plot comparison

def tmp_get_embedding(version, spectra, config):    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config['version'] = version
    model, model_config = load_trained_model_from_artifact(config)
    
    model.to(device)
    embedding = model(spectra.float().to(device)).detach().cpu().numpy()
    
    if embedding.shape[1] > 2: embedding = reduce_dimension(embedding, 2, 'pca')

    return embedding

def compute_water_gradient_vector(extended_water_timestamp, n_samples = -1):
    if(n_samples <= 0): n_samples = len(extended_water_timestamp)
    
    # Create water gradient vector
    water_gradient = np.zeros(n_samples)
    for i in range(len(water_gradient)):
        if(i == 0): pass
        if(extended_water_timestamp[i] != 0): water_gradient[i] = 0
        elif(extended_water_timestamp[i] == 0): water_gradient[i] = water_gradient[i - 1] + 1
    
    water_gradient[0] = water_gradient[1]

    # Rescale between 0 and 1
    water_gradient /= np.max(water_gradient)
    
    return water_gradient

embedding_2_3  = tmp_get_embedding('v0', spectra, model_config)
embedding_2_6  = tmp_get_embedding('v3', spectra, model_config)
embedding_16_3 = tmp_get_embedding('v1', spectra, model_config)
embedding_16_6 = tmp_get_embedding('v2', spectra, model_config)


#%%

config = dict(
    idx_list = np.arange(25) + 1,
    path_weight = 'artifacts/SpectraEmbedder_skipGram_ns_trained-v2/model_',
    dimensionaly_reduction_method = 'pca',
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    figsize = (30, 20),
    color = compute_water_gradient_vector(data[3]),
    # color = temperature,
    # color = humidity,
    # color = np.linspace(0, 1, len(temperature)),
    s = 2,
    cmap = 'autumn_r',    
)

fig, ax = plt.subplots(2,2, figsize = config['figsize'])

sc = ax[0,0].scatter(embedding_2_3[:,0], embedding_2_3[:,1], c = config['color'], s = config['s'],  cmap = config['cmap'])
sc = ax[0,1].scatter(embedding_2_6[:,0], embedding_2_6[:,1], c = config['color'], s = config['s'],  cmap = config['cmap'])
sc = ax[1,0].scatter(embedding_16_3[:,0], embedding_16_3[:,1], c = config['color'], s = config['s'],  cmap = config['cmap'])
sc = ax[1,1].scatter(embedding_16_6[:,0], embedding_16_6[:,1], c = config['color'], s = config['s'],  cmap = config['cmap'])

# ax[0,0].set_xlim([-0.16, -0.07])
# ax[0,1].set_xlim([0.12, 0.21])
# ax[1,0].set_xlim([-0.15, 0.15])
# ax[1,1].set_xlim([-0.2, 0.10])

# ax[0,0].set_ylim([-0.375, -0.175])
# ax[0,1].set_ylim([-0.46, -0.26])
# ax[1,0].set_ylim([-0.03, 0.08])
# ax[1,1].set_ylim([-0.03, 0.08])

ax[0,0].set_title('Embedding size = 2, window size = 3')
ax[0,1].set_title('Embedding size = 2, window size = 6')
ax[1,0].set_title('Embedding size = 16, window size = 3')
ax[1,1].set_title('Embedding size = 16, window size = 6')

fig.subplots_adjust(right = 0.87)
cbar_ax = fig.add_axes([0.9, 0.1, 0.05, 0.8])
fig.colorbar(sc, cax=cbar_ax)

fig.suptitle('Color based on WATER', fontsize = 50)
params = {'mathtext.default': 'regular', 'font.size': 24}       
plt.rcParams.update(params)

# plt.tight_layout()

#%% Correlation

idx = 777
signal_length_list = [10, 60, int(24 * 60)]
corr = []
n_el = 20

for signal_length in signal_length_list:
    mean_corr = np.zeros((4,4))
    
    for i in range(n_el):
        idx = np.random.randint(0, spectra.shape[0])
        tmp_humidity = humidity[idx:idx + signal_length]
        tmp_temperature = temperature[idx:idx + signal_length]
        
        mems_1 = spectra[idx:idx + signal_length, 0:300].min(1)[0].numpy()
        mems_2 = spectra[idx:idx + signal_length, -401:-1].min(1)[0].numpy()
        
        data_list = [tmp_humidity, tmp_temperature, mems_1, mems_2]
        
        tmp_corr = np.corrcoef(data_list)
        mean_corr += tmp_corr
    
    print(corr)
    corr.append(mean_corr/n_el)

#%% End file