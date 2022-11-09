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
import pickle

from support.VAE import SpectraVAE_Double_Mems, AttentionVAE
from support.VAE_Conv import SpectraVAE_Double_Mems_Conv
from support.embedding_spectra import skipGramEmbedder, CBOW
from support.embedding_sequence import Sequence_embedding_clf, SequenceEmbedderAutoencoder
from support.datasets import load_spectra_data, load_water_data, create_extended_water_vector
from support.datasets import PytorchDatasetPlantSpectra_V1, SpectraSequenceDataset
from support.preprocess import aggregate_HT_data_V2, choose_spectra_based_on_water_V1

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
        
        return model
        
        
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

#%% Build model Spectra Embedder

def build_and_log_spectra_embedder_NLP(project_name, config):
    # Get the name for the actual run
    if "skipGram" in config['type_embedder']:  run_name = get_run_name('build-SE-skipgram-embedding')
    elif "CBOW" in config['type_embedder']: run_name = get_run_name('build-SE-CBOW-embedding')
    else: raise ValueError("Problem with the type of model you want to build")
    
    with wandb.init(project = project_name, job_type = "model_creation", config = config, name = run_name) as run:
        config = wandb.config
        
        model, model_name, model_description = buld_spectra_embedder_NLP(config)
        
        # Create the artifacts
        metadata = dict(config)
        model_artifact = wandb.Artifact(model_name, type = "model", description = model_description, metadata = metadata)

        # Save the model and log it on wandb
        add_model_to_artifact(model, model_artifact, "TMP_File/untrained.pth")
        args = (torch.ones((1, config['input_size'])))
        add_onnx_to_artifact(model, model_artifact, args, "TMP_File/untrained.onnx")
        run.log_artifact(model_artifact)
        
        return model
    
def buld_spectra_embedder_NLP(config):
    model_name = "SpectraEmbedder_" + config['type_embedder']
    
    model_description = "Untrained spectra Embedder with {}. ".format(config['type_embedder'].upper())
    
    if "skipGram" in config['type_embedder']:   model = skipGramEmbedder(config)
    elif "CBOW" in config['type_embedder']:  model = CBOW(config)
   
    print(model_description)
    
    return model, model_name, model_description

    
#%% Build model Sequence Embedder

def build_and_log_Sequence_Embedder_clf_model(project_name, config):
    with wandb.init(project = project_name, job_type = "model_creation", config = config) as run:
        config = wandb.config
        
        model, model_name, model_description = build_Sequence_Embedder_clf_model(config)
        
        # Create the artifacts
        metadata = dict(config)
        model_artifact = wandb.Artifact(model_name, type = "model", description = model_description, metadata = metadata)

        # Save the model and log it on wandb
        add_model_to_artifact(model, model_artifact, "TMP_File/untrained.pth")
        add_sequence_embedder_onnx_to_artifact(config, model, model_artifact, "TMP_File/untrained.onnx")
        run.log_artifact(model_artifact)
        
        return model

def build_Sequence_Embedder_clf_model(config):
    model_name = "SequenceEmbedder_clf"
    model_description = "Untrained sequence Embedder with CLASSIFIER. "
    if config['use_spectra_embedder']: model_description += " Spectra embedder is used. "
    if config['use_attention']: model_description += " Multihead attention is used. "
    
    model = Sequence_embedding_clf(config)
    
    print(model_description)
    
    return model, model_name, model_description


def build_and_log_Sequence_Embedder_autoencoder_model(project_name, config):
    with wandb.init(project = project_name, job_type = "model_creation", config = config) as run:
        config = wandb.config
        
        model, model_name, model_description = build_Sequence_Embedder_autoencoder_model(config)
        
        # Create the artifacts
        metadata = dict(config)
        model_artifact = wandb.Artifact(model_name, type = "model", description = model_description, metadata = metadata)

        # Save the model and log it on wandb
        add_model_to_artifact(model, model_artifact, "TMP_File/untrained.pth")
        add_sequence_embedder_onnx_to_artifact(config, model, model_artifact, "TMP_File/untrained.onnx")
        run.log_artifact(model_artifact)
        
        return model


def build_Sequence_Embedder_autoencoder_model(config):
    model_name = "SequenceEmbedder_AE"
    model_description = description_Sequence_Embedder_autoencoder(config)
    
    model = SequenceEmbedderAutoencoder(config)
    # print(model_description)
    
    return model, model_name, model_description

def description_Sequence_Embedder_autoencoder(config):
    model_description = "Untrained sequence Embedder with AUTOENCODER. "
    model_description += "Embedding size = {}. ".format(config['embedder_config']['sequence_embedding_size'])
    
    # Check if spectra embedder is used
    if config['embedder_config']['use_spectra_embedder']: model_description += " USE Spectra embedder ({}). ".format(config['embedder_config']['query_embedding_size'])
    else: model_description += " NO Spectra embedder. "
    
    # Check if attention is used
    if config['embedder_config']['use_attention']: model_description += " USE Multihead attention. "
    else:  model_description += " NO Multihead attention. "
    
    # Check if LSTM in encoder use bias
    model_description += " LSTM bias encoder = {}.\n".format(config['embedder_config']['LSTM_bias'])
    
    model_description += " Decoder type = {}. ".format(config['decoder_config']['decoder_type'])
    model_description += " Decoder LSTM output size = {}. ".format(config['decoder_config']['decoder_LSTM_output_size'])
    
    return model_description


def add_sequence_embedder_onnx_to_artifact(config, model, artifact, model_name = "model.onnx"):
    if 'sequence_length' not in config: config['sequence_length'] = 5
    args = torch.rand((3, config['sequence_length'], 702))
    add_onnx_to_artifact(model, artifact, args, model_name)

#%% Load model from/to artifact

def add_model_to_artifact(model, artifact, model_name = "model.pth"):
    torch.save(model.state_dict(), model_name)
    artifact.add_file(model_name)
    wandb.save(model_name)
    

def add_onnx_to_artifact(model, artifact, args, model_name = "model.onnx"):
    torch.onnx.export(model, args = args, f = model_name, opset_version = 11)
    
    artifact.add_file(model_name)
    wandb.save(model_name)

def load_untrained_model_from_artifact(artifact_name, version = 'latest', model_name = "model.pth"):
    run = wandb.init()
    
    return load_untrained_model_from_artifact_inside_run(run, artifact_name, version, model_name)

def load_untrained_model_from_artifact_inside_run(run, artifact_name, version = 'latest', model_name = "model.pth"):
    """
    Function used to load the model of an artifact. Used inside an active run of wandb
    """
            
    model_artifact = run.use_artifact("{}:{}".format(artifact_name, version))
    model_dir = model_artifact.download()
    model_path = os.path.join(model_dir, model_name)
    model_config = model_artifact.metadata
    
    # Check which model is loaded
    # N.b. The model is selected thorugh the ARTIFACT NAME
    if "VAE" in artifact_name:
        model, model_name, model_description = build_VAE_model(model_config)
    elif "SequenceEmbedder_clf" in artifact_name:
        model, model_name, model_description = build_Sequence_Embedder_clf_model(model_config)
    elif "SequenceEmbedder_AE" in artifact_name:
        model, model_name, model_description = build_Sequence_Embedder_autoencoder_model(model_config)
    elif "skipgram" or "CBOW" in artifact_name:
        model, model_name, model_description = buld_spectra_embedder_NLP(model_config)
    else:
        raise ValueError("Problem with the type of model you want to load")

    model.load_state_dict(torch.load(model_path, map_location = torch.device('cpu')))
    
    return model, model_config

def load_trained_model_from_artifact(config):
    run = wandb.init()
    
    return load_trained_model_from_artifact_inside_run(config, run)

def load_trained_model_from_artifact_inside_run(config, run):
    if config['epoch_of_model'] < 0: config['epoch_of_model'] = 'END'
    model_name = config['model_file_name'] + '_' + str(config['epoch_of_model']) + '.pth'
    
    # Retrieve trained model weights
    model_artifact = run.use_artifact("{}:{}".format(config['artifact_name'], config['version']))
    model_dir = model_artifact.download()
    model_path = os.path.join(model_dir, model_name)
    model_config = model_artifact.metadata['model_config']
    print(model_config)
    
    # Create model and load the weights
    if "SpectraVAE_" in config['artifact_name']:
        model, model_name, model_description = build_VAE_model(model_config)
    elif "SequenceEmbedder_clf" in config['artifact_name']:
        model, model_name, model_description = build_Sequence_Embedder_clf_model(model_config)
    elif "SequenceEmbedder_AE" in config['artifact_name']:
        model, model_name, model_description = build_Sequence_Embedder_autoencoder_model(model_config)
    else:
        raise ValueError("Problem with the type of model you want to load")

    # model, model_name, model_description = build_VAE_model(model_config)
    model.load_state_dict(torch.load(model_path, map_location = torch.device('cpu')))
    
    # Retrieve index
    a_file = open(os.path.join(model_dir, "idx_dict.pkl"), "rb")
    idx_dict = pickle.load(a_file)
    a_file.close()
    
    return model, model_config, idx_dict

# TODO
# def load_SE_AE_model_from_artifact_inside_run(config, run):
#     if config['epoch_of_model'] < 0: config['epoch_of_model'] = 'END'
#     model_name = config['model_file_name'] + '_' + config['epoch_of_model'] + '.pth'

#%% Dataset

def load_dataset_local_VAE(config):
    # Sepctra
    spectra_plants_numpy, wavelength, timestamp = load_spectra_data("data/[2021-08-05_to_11-26]All_PlantSpectra.csv", config['normalize_trials'])

    # Water
    water_data, water_timestamp = load_water_data("data/[2021-08-05_to_11-26]PlantTest_Notes.csv")
    extended_water_timestamp = create_extended_water_vector(water_timestamp, water_data, timestamp)

    # Divide the spectra in good (Water) and Bad (NON water)
    good_idx, bad_idx = choose_spectra_based_on_water_V1(extended_water_timestamp, time_interval_start = config['time_interval_start'], time_interval_end = config['time_interval_end'])
    
    good_spectra_dataset = PytorchDatasetPlantSpectra_V1(spectra_plants_numpy[good_idx, :], used_in_cnn = config['use_cnn'])
    bad_spectra_dataset = PytorchDatasetPlantSpectra_V1(spectra_plants_numpy[bad_idx, :], used_in_cnn = config['use_cnn'])
    
    return good_spectra_dataset, bad_spectra_dataset, good_idx, bad_idx

def load_dataset_local_Sequence_embedder_clf(load_config, dataset_config):
    data = load_dataset_from_artifact(load_config)
    
    spectra = data[0]
    h_array = data[4]
    
    dataset = SpectraSequenceDataset(spectra, h_array, dataset_config)
    
    return dataset


def split_data(data, config):
    if 'print_var' not in config: config['print_var'] = True
    
    percentage_train, percentage_test, percentage_validation = config['split_percentage_list']
    if percentage_train + percentage_test + percentage_validation > 1:
        print(percentage_train + percentage_test + percentage_validation)
        raise ValueError("The sum of the percentage of train, test and validation must be lower or equal to 1")

    # Create index to divide the dataset in 3 (train, test and validation)
    idx = np.arange(data.shape[0])
    tmp_len_list = (np.asarray(config['split_percentage_list']) * data.shape[0]).astype(int)
    tmp_len_list[-1] = abs(data.shape[0] - tmp_len_list.sum()) + tmp_len_list[-1]
    train_idx, test_idx, validation_idx = torch.utils.data.random_split(idx, tmp_len_list)
    
    idx_list = [train_idx, test_idx, validation_idx]
     
    if config['print_var']:
        print("Length Training set   = " + str(len(train_idx)))
        print("Length Test set       = " + str(len(test_idx)))
        print("Length Validation set = " + str(len(validation_idx)))
        
    return train_idx, test_idx, validation_idx, idx_list


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


def load_dataset_from_artifact(config):
    """
    Create a wandb run and load the dataset artifact
    """
    run = wandb.init()
    
    return load_dataset_from_artifact_inside_run(config, run)
        
def load_dataset_from_artifact_inside_run(config, run):
    """
    Load the spectra data (and eventual the other sensor data) from a wandb artifact inside a run
    """
    print("{}:{}".format(config['artifact_name'], config['version']))
    dataset_artifact = run.use_artifact("{}:{}".format(config['artifact_name'], config['version']), type='dataset')
    dataset_dir = dataset_artifact.download()
    dataset_path = os.path.join(dataset_dir, config['spectra_file_name'])

    spectra_plants_numpy, wavelength, spectra_timestamp = load_spectra_data(dataset_path, config['normalize_trials'])
    
    if config['return_other_sensor_data']:
        # Water data
        water_path = os.path.join(dataset_dir, config['water_file_name'])
        water_data, water_timestamp = load_water_data(water_path)
        extended_water_timestamp = create_extended_water_vector(water_timestamp, water_data, spectra_timestamp)
        
        # HT data
        a = load_ht_data(os.path.join(dataset_dir, config['ht_file_path']), os.path.join(dataset_dir, config['ht_timestamp_path']), os.path.join(dataset_dir, config['spectra_timstamp_path']))
        aggregate_h_array, aggregate_t_array, aggregate_timestamp = a[0], a[1], a[2]
        
        return spectra_plants_numpy, wavelength, spectra_timestamp, extended_water_timestamp, aggregate_h_array, aggregate_t_array
    else:
        return spectra_plants_numpy, wavelength, spectra_timestamp
            
        
def load_ht_data(ht_file_path, ht_timestamp_path, spectra_timstamp_path):
    """
    Load the data of the HT sensor. Then the data are aggregate in order to have the same timestamp of the spectra.
    ht_file_path:           path of the file with the ht data
    ht_timestamp_path:      path of the file with the timestamp of the ht data
    spectra_timstamp_path:  path of the file with the spectra timestamp
    """
    
    ht_data = pd.read_csv(ht_file_path, encoding= 'unicode_escape')
    h_array = ht_data[' Humidity[%]'].to_numpy()
    t_array = ht_data[' Temperature[C]'].to_numpy()
    
    ht_timestamp = pd.read_csv(ht_timestamp_path).to_numpy()[:, 1:]
    spectra_timestamp = pd.read_csv(spectra_timstamp_path).to_numpy()[:, 1:]
    
    a = aggregate_HT_data_V2(ht_timestamp, spectra_timestamp, h_array, t_array)
    aggregate_h_array, aggregate_t_array, aggregate_timestamp = a[0], a[1], a[2]
    
    return aggregate_h_array, aggregate_t_array, aggregate_timestamp
        
def load_dataset_h_local(config):
    """
    Load and divide the dataset based on humidity data
    """
    # Sepctra
    spectra_plants_numpy, wavelength, timestamp = load_spectra_data("data/[2021-08-05_to_11-26]All_PlantSpectra.csv", config['normalize_trials'])
    
    # Load the ht data (divided in 2 rows so it is easy to read)
    a = load_ht_data("data/[2021-08-05_to_11-26]All_PlantHTSensor.csv", 'data/jesus_ht_timestamp.csv', 'data/jesus_spectra_timestamp.csv')
    aggregate_h_array, aggregate_t_array, aggregate_timestamp = a[0], a[1], a[2]
    
    good_idx = aggregate_h_array < np.mean(aggregate_h_array)
    bad_idx = aggregate_h_array >= np.mean(aggregate_h_array)
    
    good_spectra_dataset = PytorchDatasetPlantSpectra_V1(spectra_plants_numpy[good_idx, :], used_in_cnn = config['use_cnn'])
    bad_spectra_dataset = PytorchDatasetPlantSpectra_V1(spectra_plants_numpy[bad_idx, :], used_in_cnn = config['use_cnn'])
    
    if config['print_var']:
        print("Length good dataset: ", len(good_spectra_dataset))
        print("Length bad dataset : ", len(bad_spectra_dataset))
    
    return good_spectra_dataset, bad_spectra_dataset

#%% Other functions

def get_run_name(run_name):
    counter_run = open('counter_run.txt', 'r')
    lines = counter_run.readlines()
      
    run_find = False
    new_run_name = ""
    tmp_string = "" # To save the file content 
    for line in lines:
        tmp_run_name = line.split(' ')[0]
        run_count = int(line.split(' ')[1])
        
        if tmp_run_name == run_name:
            run_find = True
            new_run_name = tmp_run_name + " " + str(run_count + 1)
            tmp_string += new_run_name + "\n"
        else:
            tmp_string += line
        
    if not run_find:
        raise ValueError("{} not find in the list inside counter_run file".format(run_name))
    else:
        counter_run.close() # Close the file
        
        # Reopen the file and write the new content
        text_file = open("counter_run.txt", "wt")
        n = text_file.write(tmp_string.strip())
        text_file.close()
        
        return new_run_name