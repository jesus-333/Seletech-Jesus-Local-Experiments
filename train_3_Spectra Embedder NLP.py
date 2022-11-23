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
import torch
import wandb

from support.wandb_init_V2 import build_and_log_spectra_embedder_NLP
from support.wandb_training_Spectra_Embedder import train_and_log_SE_model

#%% Wandb login and log file inizialization

wandb.login()

project_name = "Seletech VAE Spectra"

#%% Build model

model_config = dict(
    input_size = 702,
    embedding_size = 32,
    type_embedder= 'skipGram_ns',
    window_size = 3, # Not used for skipGram_ns
    debug = False
    )

untrained_mode = build_and_log_spectra_embedder_NLP(project_name, model_config)

#%% Train model

dataset_config = dict(
    # Artifacts info
    artifact_name = 'jesus_333/Seletech VAE Spectra/Dataset_Spectra_1',
    version = 'latest',
    return_other_sensor_data = False,
    spectra_file_name = '[2021-08-05_to_11-26]All_PlantSpectra.csv',
    water_file_name = '[2021-08-05_to_11-26]PlantTest_Notes.csv',
    ht_file_path = '[2021-08-05_to_11-26]All_PlantHTSensor.csv',
    ht_timestamp_path = 'jesus_ht_timestamp.csv', 
    spectra_timstamp_path = 'jesus_spectra_timestamp.csv',
    # Normalization settings
    normalize_trials = 2,
    # Dataset config
    window_size = 6
)

train_config = dict(
    model_artifact_name = 'SpectraEmbedder_skipGram_ns',
    version = 'v0', # REMEMBER ALWAYS TO CHECK THE VERSION
    # Numerical Hyperparameter
    batch_size = 32,
    lr = 1e-3,
    epochs = 25,
    use_scheduler = True,
    gamma = 0.9, # Parameter of the lr exponential scheduler
    optimizer_weight_decay = 1e-2,
    num_negative_sample = 5,
    # Support stuff (device, log frequency etc)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    log_freq = 1,
    epoch_to_save_model = 1,
    dataset_config = dataset_config,
    print_var = True,
    debug = True,
)

trained_model = train_and_log_SE_model(project_name, train_config)

#%% TMP PLot embedding
import torch
import wandb
import matplotlib.pyplot as plt
import numpy as np
from support.wandb_init_V2 import load_trained_model_from_artifact, load_untrained_model_from_artifact
from support.wandb_init_V2 import load_dataset_from_artifact
from support.datasets import SpectraNLPDataset
from support.wandb_visualization import reduce_dimension
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def compute_embedding(embedder, dataset, config):
    embedder.to(config['device'])
    embedding_list = []
    
    for i in range(len(dataset)):
        # Move data to device
        x = dataset[i][0].to(config['device'])
        
        # Compute the embedding
        # TODO Implement CBOW version. For now works only for skipGram
        if 'CBOW' in str(type(embedder)): raise ValueError("The embedding for now works only for skipGram")
        tmp_emb = embedder(x)
    
        # Save the results for the batch
        embedding_list.append(tmp_emb.detach().cpu().unsqueeze(0))
    
    # Convert embedding list in a single numpy array
    embedding = torch.cat(embedding_list).numpy()
    

    return embedding


def compute_water_gradient_vector(extended_water_timestamp, n_samples):
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
    normalize_trials = 1,
    # Dataset config
    window_size = 2,
    batch_size = 32
)

data = load_dataset_from_artifact(dataset_config)

# dataset = SpectraSequenceDataset(data[0], dataset_config, data[4])
dataset = SpectraNLPDataset(data[0], dataset_config)

model_config = dict(
    artifact_name = 'jesus_333/Seletech VAE Spectra/SpectraEmbedder_skipGram_ns_trained',
    version = 'v2',
    epoch_of_model = 3,
    model_file_name = 'model'
)

model, model_config = load_trained_model_from_artifact(model_config)

color = compute_water_gradient_vector(data[3], -1)
config = dict(
    figsize = (10, 8),
    s = 2,
    cmap = 'Greens_r'
)
color = color[dataset_config['window_size']:-dataset_config['window_size']]
# color = data[5][dataset_config['window_size']:-dataset_config['window_size']]

fig_list = []
ax_list = []

for i in range(1, 101):
    print(i)
    model_path = 'artifacts/SpectraEmbedder_skipGram_trained-v2/model_{}.pth'.format(i)
    model.load_state_dict(torch.load(model_path, map_location = torch.device('cpu')))
    
    plot_config = dict(
        input_type = 'spectra_embedding',
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    )

    embedding = compute_embedding(model.embedder.to(device), dataset, plot_config)
    

    fig, ax = plt.subplots(figsize = config['figsize'])
    sc = ax.scatter(embedding[:,0], embedding[:,1], c = color, s = config['s'],  cmap = config['cmap'])
    plt.colorbar(sc)
    
    fig_list.append(fig)
    ax_list.append(ax)
    
    plt.show()

#%%

for i in range(50,100):
    fig = fig_list[i]
    ax = ax_list[i]
    ax.relim()          # make sure all the data fits
    ax.autoscale(True)  # auto-scale
    ax.figure.canvas.draw_idle()    
    fig.savefig('TMP_File/Plot/tmp_{}.png'.format(i), bbox_inches='tight')
    
#%% End file 
