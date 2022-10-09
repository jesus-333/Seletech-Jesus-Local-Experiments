"""
Created on Fri Sep 23 15:16:29 2022

@author: Alberto Zancanaro (jesus)
@organization: University of Padua
"""

#%% Imports

import numpy as np
import torch
import wandb
import os
import pandas as pd

from support.VAE import SpectraVAE_Double_Mems, AttentionVAE
from support.VAE_Conv import SpectraVAE_Double_Mems_Conv
from support.embedding_sequence import SequenceEmbedder_V2
from support.datasets import load_spectra_data, load_water_data, create_extended_water_vector, choose_spectra_based_on_water_V1
from support.datasets import PytorchDatasetPlantSpectra_V1
from support.preprocess import aggregate_HT_data_V2

#%% Build model VAE/AE

def build_and_log_VAE_model(project_name, config):
    if config['neurons_per_layer'][1] / config['neurons_per_layer'][0] != 2 and config['use_cnn'] == False:
        raise ValueError("FOR NOW, the number of neurons of the second hidden layer must be 2 times the number of neurons in the first inner layer")
        
    with wandb.init(project = project_name, job_type = "model_creation", config = config) as run:
        config = wandb.config
        
        model, model_name, model_description = build_VAE_model(config)
        
        # Create the artifacts
        metadata = dict(config)
        model_artifact = wandb.Artifact(model_name, type = "model", description = model_description, metadata = metadata)

        # Save the model and log it on wandb
        add_model_to_artifact(model, model_artifact, "TMP_File/untrained.pth")
        args = (torch.ones((1, 300)), torch.ones((1, 400)))
        add_onnx_to_artifact(model, model_artifact, args, "TMP_File/untrained.onnx")
        run.log_artifact(model_artifact)
        
        
def build_VAE_model(config):
    # Create the model
    if config['use_cnn']: # Convolutional VAE 
        model = SpectraVAE_Double_Mems_Conv(config['length_mems_1'], config['length_mems_2'], config['hidden_space_dimension'], 
                                            use_as_autoencoder = config['use_as_autoencoder'], use_bias = config['use_bias'],
                                            print_var = config['print_var'])
        if config['use_as_autoencoder']: 
            model_name = "SpectraAE_CNN"
            model_description = "Untrained AE model. Convolutional version. Hidden space = {}".format(config['hidden_space_dimension'])
        else: 
            model_name = "SpectraVAE_CNN"
            model_description = "Untrained VAE model. Convolutional version. Hidden space = {}".format(config['hidden_space_dimension'])
        
    else:
        if config['use_attention']: # Feed-Forward VAE with attention. ATTENTION NOT WORK FOR NOW
            model = AttentionVAE(config['length_mems_1'], config['length_mems_2'], 
                               config['hidden_space_dimension'], config['embedding_size'],
                               print_var = config['print_var'], use_as_autoencoder = config['use_as_autoencoder'] )
            model_name = "SpectraVAE_FC_Attention"
            model_description = "Untrained VAE model. Fully-connected version + Attention. Hidden space = {}".format(config['hidden_space_dimension'])
        else: # Feed-Forward VAE without attention
            model = SpectraVAE_Double_Mems(config['length_mems_1'], config['length_mems_2'],  config['neurons_per_layer'], config['hidden_space_dimension'], 
                                         use_as_autoencoder = config['use_as_autoencoder'], use_bias = config['use_bias'],
                                         print_var = config['print_var'])
            if config['use_as_autoencoder']: 
                model_name = "SpectraAE_FC"
                model_description = "Untrained AE model. Fully-connected version. Hidden space = {}".format(config['hidden_space_dimension'])
            else: 
                model_name = "SpectraVAE_FC"
                model_description = "Untrained VAE model. Fully-connected version. Hidden space = {}".format(config['hidden_space_dimension'])
                print(model_description)
            
    return model, model_name, model_description

#%% Build model Sequence Embedder

def build_and_log_Sequence_Embedder_model(project_name, config):
    with wandb.init(project = project_name, job_type = "model_creation", config = config) as run:
        config = wandb.config
        
        model, model_name, model_description = build_Sequence_Embedder_model(config)
        
        # Create the artifacts
        metadata = dict(config)
        model_artifact = wandb.Artifact(model_name, type = "model", description = model_description, metadata = metadata)

        # Save the model and log it on wandb
        add_model_to_artifact(model, model_artifact, "TMP_File/untrained.pth")
        add_sequence_embedder_onnx_to_artifact(config, model, model_artifact, "TMP_File/untrained.onnx")
        run.log_artifact(model_artifact)

def build_Sequence_Embedder_model(config):
    model_name = "SequenceEmbedder"
    model_description = "Untrained sequence Embedder. "
    if config['use_spectra_embedder']: model_description += " Spectra embedder is used. "
    if config['use_attention']: model_description += " Multihead attention is used. "
    
    model = SequenceEmbedder_V2(config)
    
    print(model_description)
    
    return model, model_name, model_description

def add_sequence_embedder_onnx_to_artifact(config, model, artifact, model_name = "model.onnx"):
    if 'sequence_length' not in config: config['sequence_length'] = 5
    args = torch.rand((1, config['sequence_length'], 700))
    add_onnx_to_artifact(model, artifact, args, model_name)

#%% Load model from/to artifact

def add_model_to_artifact(model, artifact, model_name = "model.pth"):
    torch.save(model.state_dict(), model_name)
    artifact.add_file(model_name)
    wandb.save(model_name)
    

def add_onnx_to_artifact(model, artifact, args, model_name = "model.onnx"):
    torch.onnx.export(model, args = args, f = model_name)
    
    artifact.add_file(model_name)
    wandb.save(model_name)

def load_model_to_artifact(artifact_name, version = 'latest', model_name = "model.pth"):
    run = wandb.init()
    
    return load_model_from_artifact_inside_run(run, artifact_name, version, model_name)

    
def load_model_from_artifact_inside_run(run, artifact_name, version = 'latest', model_name = "model.pth"):
    """
    Function used to load the model of an artifact. Used inside an active run of wandb
    """
    model_artifact = run.use_artifact("{}:{}".format(artifact_name, version))
    model_dir = model_artifact.download()
    model_path = os.path.join(model_dir, model_name)
    model_config = model_artifact.metadata
    
    model, model_name, model_description = build_VAE_model(model_config)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    return model, model_config
    

#%% Dataset

def load_dataset_local(config):
    # Sepctra
    spectra_plants_numpy, wavelength, timestamp = load_spectra_data("data/[2021-08-05_to_11-26]All_PlantSpectra.csv", config['normalize_trials'])

    # Water
    water_data, water_timestamp = load_water_data("data/[2021-08-05_to_11-26]PlantTest_Notes.csv")
    extended_water_timestamp = create_extended_water_vector(water_timestamp, water_data, timestamp)

    # Divide the spectra in good (Water) and Bad (NON water)
    good_idx, bad_idx = choose_spectra_based_on_water_V1(extended_water_timestamp, time_interval_start = config['time_interval_start'], time_interval_end = config['time_interval_end'])
    
    good_spectra_dataset = PytorchDatasetPlantSpectra_V1(spectra_plants_numpy[good_idx, :], used_in_cnn = config['use_cnn'])
    bad_spectra_dataset = PytorchDatasetPlantSpectra_V1(spectra_plants_numpy[bad_idx, :], used_in_cnn = config['use_cnn'])
    
    return good_spectra_dataset, bad_spectra_dataset


def split_dataset(dataset, config):
    if 'print_var' not in config: config['print_var'] = True
    
    percentage_train, percentage_test, percentage_validation = config['split_percentage_list']
    if percentage_train + percentage_test + percentage_validation > 1:
        print(percentage_train + percentage_test + percentage_validation)
        raise ValueError("The sum of the percentage of train, test and validation must be lower or equal to 1")
            
    dataset_train = dataset[0:int(len(dataset) * percentage_train)]
    dateset_test = dataset[int(len(dataset) * percentage_train):int(len(dataset) * (percentage_train + percentage_test))]
    dataset_validation = dataset[int(len(dataset) * (percentage_train + percentage_test)):]
    
    if config['print_var']:
        print("Length Training set   = " + str(len(dataset_train)))
        print("Length Test set       = " + str(len(dateset_test)))
        print("Length Validation set = " + str(len(dataset_validation)))
        
    return dataset_train, dateset_test, dataset_validation


def log_data(project_name):
    """
    Create the artifact with the data of the first acquisition campaing
    """
    with wandb.init(project = project_name, job_type = "Load_dataset") as run:
        dataset_description = 'Artifact with the raw data of the first campaign. Contain the plants spectra, the water info, the HT sensor data and the MIFlower data'
        data_artifact = wandb.Artifact("Dataset_Spectra_1", type = "dataset", description = dataset_description)
        
        data_artifact.add_file("data/[2021-08-05_to_11-26]All_PlantSpectra.csv")
        data_artifact.add_file("data/[2021-08-05_to_11-26]PlantTest_Notes.csv")
        data_artifact.add_file("data/[2021-08-05_to_11-26]All_PlantHTSensor.csv")
        data_artifact.add_file("data/[2021-08-05_to_11-26]All_PlantMiFlowerCareSensor.csv")
        data_artifact.add_file("data/jesus_ht_timestamp.csv")
        data_artifact.add_file("data/jesus_spectra_timestamp.csv")
        
        run.log_artifact(data_artifact)
        
        
def load_dataset_h_local(config):
    """
    Load and divide the dataset based on humidity data
    """
    # Sepctra
    spectra_plants_numpy, wavelength, timestamp = load_spectra_data("data/[2021-08-05_to_11-26]All_PlantSpectra.csv", config['normalize_trials'])

    ht_data = pd.read_csv("data/[2021-08-05_to_11-26]All_PlantHTSensor.csv", encoding= 'unicode_escape')
    h_array = ht_data[' Humidity[%]'].to_numpy()
    t_array = ht_data[' Temperature[C]'].to_numpy()
    
    ht_timestamp = pd.read_csv('data/jesus_ht_timestamp.csv').to_numpy()[:, 1:]
    spectra_timestamp = pd.read_csv('data/jesus_spectra_timestamp.csv').to_numpy()[:, 1:]
    
    a = aggregate_HT_data_V2(ht_timestamp, spectra_timestamp, h_array, t_array)
    aggregate_h_array, aggregate_t_array, aggregate_timestamp = a[0], a[1], a[2]
    
    good_idx = aggregate_h_array < np.mean(aggregate_h_array)
    bad_idx = aggregate_h_array >= np.mean(aggregate_h_array)
    
    good_spectra_dataset = PytorchDatasetPlantSpectra_V1(spectra_plants_numpy[good_idx, :], used_in_cnn = config['use_cnn'])
    bad_spectra_dataset = PytorchDatasetPlantSpectra_V1(spectra_plants_numpy[bad_idx, :], used_in_cnn = config['use_cnn'])
    
    if config['print_var']:
        print("Length good dataset: ", len(good_spectra_dataset))
        print("Length bad dataset : ", len(bad_spectra_dataset))
    
    return good_spectra_dataset, bad_spectra_dataset