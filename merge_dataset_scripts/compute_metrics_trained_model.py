"""
Test the dataset and the model for the training with merged data
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
import json
import torch
import numpy as np

from library import datasets, HydraNet, metrics

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

path_saved_model = 'saved model/merged_data/'
n_model = 10

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def compute_and_save_metrics(model, loader, train_config, tmp_log_dict, data_type : str) :
    # Get the dataset
    dataset = loader.dataset
    x_mems_1, x_mems_2, true_labels, labels_text, source_array = dataset[:]

    # Compute the Accuracy
    x_mems_1, x_mems_2 = x_mems_1.to(train_config['device']), x_mems_2.to(train_config['device'])
    metrics_per_head_list = model.compute_metrics_batch(x_mems_1, x_mems_2, source_array, true_labels)

    for i in range(len(metrics_per_head_list)):
        metrics_per_head = metrics_per_head_list[i]
        for metric in metrics_per_head :
            tmp_log_dict['{} head {} ({})'.format(metric, model.head_sources[i], data_type)] = metrics_per_head[metric]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Get config
config = json.load(open('training_scripts/config/config_1.json', 'r'))
train_config = config['training_config']

# Get dataset
full_dataset = datasets.NIRS_dataset_merged(config['training_config']['source_path_list'])

# Divide the dataset in training and testing
idx_train, idx_test = datasets.get_idx_to_split_data(full_dataset.data_mems_1.shape[0], config['training_config']['percentage_split_train_test'], config['training_config']['seed'])
idx_train, idx_val = datasets.get_idx_to_split_data(len(idx_train), config['training_config']['percentage_split_train_validation'], config['training_config']['seed'], idx_train)
train_dataset = datasets.NIRS_dataset_merged(config['training_config']['source_path_list'], idx_train)
test_dataset = datasets.NIRS_dataset_merged(config['training_config']['source_path_list'], idx_test)
validation_dataset = datasets.NIRS_dataset_merged(config['training_config']['source_path_list'], idx_val)

# Create DataLoader
train_loader      = torch.utils.data.DataLoader(train_dataset,      batch_size = config['training_config']['batch_size'], shuffle = True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size = config['training_config']['batch_size'], shuffle = True)
test_loader       = torch.utils.data.DataLoader(test_dataset,       batch_size = config['training_config']['batch_size'], shuffle = True)

# Update model config and create model
config['model_config']['config_body']['input_size_mems_1'] = full_dataset.data_mems_1.shape[1]
config['model_config']['config_body']['input_size_mems_2'] = full_dataset.data_mems_2.shape[1]
model = HydraNet.hydra_net_v1(config['model_config']['config_body'], config['model_config']['config_heads'])
config['training_config']['device'] = 'cpu'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Compute metrics

list_metrics_END = []
list_metrics_BEST = []
for i in range(n_model) :
    print(i)

    tmp_log_dict_END = {}
    tmp_log_dict_BEST = {}

    path_saved_model_END = path_saved_model + '{}/model_100.pth'.format(i + 1)
    path_saved_model_BEST = path_saved_model + '{}/model_BEST.pth'.format(i + 1)

    model.load_state_dict(torch.load(path_saved_model_END, map_location = 'cpu'))
    compute_and_save_metrics(model, train_loader, train_config, tmp_log_dict_END, 'TRAIN')
    compute_and_save_metrics(model, validation_loader, train_config, tmp_log_dict_END, 'VALIDATION')
    compute_and_save_metrics(model, test_loader, train_config, tmp_log_dict_END, 'TEST')
    list_metrics_END.append(tmp_log_dict_END)

    model.load_state_dict(torch.load(path_saved_model_BEST, map_location = 'cpu'))
    compute_and_save_metrics(model, train_loader, train_config, tmp_log_dict_BEST, 'TRAIN')
    compute_and_save_metrics(model, validation_loader, train_config, tmp_log_dict_BEST, 'VALIDATION')
    compute_and_save_metrics(model, test_loader, train_config, tmp_log_dict_BEST, 'TEST')
    list_metrics_BEST.append(tmp_log_dict_BEST)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Print results

string_to_print_TRAIN_END = metrics.compute_average_and_std(list_metrics_END, 'TRAIN')
string_to_print_VALIDATION_END = metrics.compute_average_and_std(list_metrics_END, 'VALIDATION')
string_to_print_TEST_END = metrics.compute_average_and_std(list_metrics_END, 'TEST')

string_to_print_TRAIN_BEST = metrics.compute_average_and_std(list_metrics_BEST, 'TRAIN')
string_to_print_VALIDATION_BEST = metrics.compute_average_and_std(list_metrics_BEST, 'VALIDATION')
string_to_print_TEST_BEST = metrics.compute_average_and_std(list_metrics_BEST, 'TEST')

