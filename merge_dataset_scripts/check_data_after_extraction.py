# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import json
import torch
import numpy as np
import pprint

from library import datasets

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def count_sources_and_labels(dataset_list, list_source_to_count, list_labels_to_count):
    # Variable to store the count
    count_source = {source: 0 for source in list_source_to_count}
    count_labels = {label : 0 for label  in list_labels_to_count}

    for dataset in dataset_list:
        # Get labels and source form the dataset
        _, _, true_labels, labels_text, source_array = dataset[:]

        # Count sources
        for source in list_source_to_count: count_source[source] += np.sum(source_array == source)

        # Count labels
        for label in list_labels_to_count: count_labels[label] += np.sum(labels_text == label)

    return count_source, count_labels

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Load the data and create DataLoader

# Get config
config = json.load(open('training_scripts/config/config_1.json', 'r'))

# Get dataset
full_dataset = datasets.NIRS_dataset_merged(config['training_config']['source_path_list'])

# Divide the dataset in training, testing and validation
idx_train, idx_test = datasets.get_idx_to_split_data(full_dataset.data_mems_1.shape[0], config['training_config']['percentage_split_train_test'], config['training_config']['seed'])
idx_train, idx_val = datasets.get_idx_to_split_data(len(idx_train), config['training_config']['percentage_split_train_validation'], config['training_config']['seed'], idx_train)

train_dataset = datasets.NIRS_dataset_merged(config['training_config']['source_path_list'], idx_train)
test_dataset = datasets.NIRS_dataset_merged(config['training_config']['source_path_list'], idx_test)
validation_dataset = datasets.NIRS_dataset_merged(config['training_config']['source_path_list'], idx_val)

# Create Dataloader
train_loader      = torch.utils.data.DataLoader(train_dataset,      batch_size = config['training_config']['batch_size'], shuffle = True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size = config['training_config']['batch_size'], shuffle = True)
test_loader       = torch.utils.data.DataLoader(test_dataset,       batch_size = config['training_config']['batch_size'], shuffle = True)

# Other configurations
device = "cuda" if torch.cuda.is_available() else "cpu"  # device (i.e. cpu/gpu) used to train the network.
# device = "cpu"
config['training_config']['device'] = device

# Save the indices for the training, validation and testing in the config
config['training_config']['idx_train'] = idx_train
config['training_config']['idx_val']   = idx_val
config['training_config']['idx_test']  = idx_test

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Count samples in the dataset before and after division in training, testing and validation

list_source_to_count = ["beans", "orange", "potos"]
list_labels_to_count = ["ViciaFaba", "PhaseolusVulgaris",'orange', 'orangeDOWN_whiteUP', 'white', 'whole_orange', 'wet', 'dry']

dataset_list = [full_dataset]
count_source, count_labels = count_sources_and_labels(dataset_list, list_source_to_count, list_labels_to_count)

# Print the count (BEFORE)
print("BEFORE division in training, testing and validation")
pprint.pprint(count_source)
pprint.pprint(count_labels)

dataset_list = [train_dataset, validation_dataset, test_dataset]
count_source, count_labels = count_sources_and_labels(dataset_list, list_source_to_count, list_labels_to_count)

# Print the count (AFTER)
print("AFTER division in training, testing and validation")
pprint.pprint(count_source)
pprint.pprint(count_labels)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Check indices

all_indices = set(list(idx_train) + list(idx_val) + list(idx_test))
print("N. of indices :" , len(all_indices))

