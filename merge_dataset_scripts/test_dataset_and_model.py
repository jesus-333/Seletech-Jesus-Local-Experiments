"""
Test the dataset and the model for the training with merged data
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
import json
import torch

from library import datasets, HydraNet

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Get config
config = json.load(open('merge_dataset_scripts/config.json', 'r'))

# Get dataset
dataset = datasets.NIRS_dataset_merged(config['training_config']['source_path_list'])

# Divide the dataset in training and testing
idx_train, idx_test = datasets.get_idx_to_split_data(dataset.data_mems_1.shape[0], config['training_config']['percentage_split_train_test'], config['training_config']['seed'])
train_dataset = torch.utils.data.Subset(dataset, idx_train)
test_dataset = torch.utils.data.Subset(dataset, idx_test)

# Divide the training dataset in training and validation
idx_train, idx_val = datasets.get_idx_to_split_data(len(train_dataset), config['training_config']['percentage_split_train_validation'], config['training_config']['seed'])
train_dataset = torch.utils.data.Subset(train_dataset, idx_train)
validation_dataset = torch.utils.data.Subset(train_dataset, idx_val)

# Update model config and create model
config['model_config']['config_body']['input_size_mems_1'] = dataset.data_mems_1.shape[1]
config['model_config']['config_body']['input_size_mems_2'] = dataset.data_mems_2.shape[1]
model = HydraNet.hydra_net_v1(config['model_config']['config_body'], config['model_config']['config_heads'])

# Test forward pass
x_mems_1, x_mems_2, labels, labels_text, source_array = train_dataset[259:297]
out = model(x_mems_1, x_mems_2, source_array)

print("Train dataset size      : ", len(train_dataset))
print("Validation dataset size : ", len(validation_dataset))
print("Test dataset size       : ", len(test_dataset))
