"""
Test the dataset and the model for the training with merged data
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
import json
from library import datasets, HydraNet

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

config = json.load(open('merge_dataset_scripts/config.json', 'r'))

dataset = datasets.NIRS_dataset_merged(config['training_config']['source_path_list'])

config['model_config']['config_body']['input_size_mems_1'] = dataset.data_mems_1.shape[1]
config['model_config']['config_body']['input_size_mems_2'] = dataset.data_mems_2.shape[1]
model = HydraNet.hydra_net_v1(config['model_config']['config_body'], config['model_config']['config_heads'])
