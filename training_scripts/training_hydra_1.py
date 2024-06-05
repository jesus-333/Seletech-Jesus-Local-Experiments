# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import json
import torch
import wandb
import os

from library import datasets, HydraNet

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Setup the training

# Get config
config = json.load(open('training_scripts/config/config_1.json', 'r'))

# Get dataset
full_dataset = datasets.NIRS_dataset_merged(config['training_config']['source_path_list'])

# Divide the dataset in training, testing and validation
idx_train, idx_test = datasets.get_idx_to_split_data(full_dataset.data_mems_1.shape[0], config['training_config']['percentage_split_train_test'], config['training_config']['seed'])
idx_train, idx_val = datasets.get_idx_to_split_data(len(idx_train), config['training_config']['percentage_split_train_validation'], config['training_config']['seed'])
train_dataset = datasets.NIRS_dataset_merged(config['training_config']['source_path_list'], idx_train)
test_dataset = datasets.NIRS_dataset_merged(config['training_config']['source_path_list'], idx_test)
validation_dataset = datasets.NIRS_dataset_merged(config['training_config']['source_path_list'], idx_val)

# Update model config and create model
config['model_config']['config_body']['input_size_mems_1'] = full_dataset.data_mems_1.shape[1]
config['model_config']['config_body']['input_size_mems_2'] = full_dataset.data_mems_2.shape[1]
model = HydraNet.hydra_net_v1(config['model_config']['config_body'], config['model_config']['config_heads'])

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

def add_file_to_artifact(artifact, file_name):
    artifact.add_file(file_name)
    wandb.save(file_name)

def compute_and_save_accuracy(model, loader, train_config, log_dict, data_type : str) :
    # Get the dataset
    dataset = loader.dataset
    x_mems_1, x_mems_2, true_labels, labels_text, source_array = dataset[:]

    # Compute the Accuracy
    x_mems_1, x_mems_2 = x_mems_1.to(train_config['device']), x_mems_2.to(train_config['device'])
    accuracy_per_head_list = model.compute_accuracy_batch(x_mems_1, x_mems_2, source_array, true_labels)

    for i in range(len(accuracy_per_head_list)):
        log_dict['Accuracy head {} ({})'.format(model.head_sources[i], data_type)] = accuracy_per_head_list[i]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

with wandb.init(project = 'Seletech-Jesus-Local-Experiments-Merge-Data', config = config) as run:

    train_config = config['training_config']
    notes = train_config['notes']
    name = train_config['name_training_run'] if 'name_training_run' in train_config else None
    train_config['wandb_training'] = True

    # Setup artifact to save model
    model_artifact_name = train_config['model_artifact_name'] + '_trained'
    metadata = dict(config)
    model_artifact = wandb.Artifact(model_artifact_name, type = "model",
                                    description = "Trained {} model".format(train_config['model_artifact_name']),
                                    metadata = metadata)
    
    # Used to save the loss and the accuracy during the training
    log_dict = {}

    # Optimizer and loss function
    loss_function = torch.nn.NLLLoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr = train_config['lr'],
                                  weight_decay = train_config['optimizer_weight_decay']
                                  )

    # (OPTIONAL) Setup lr scheduler
    if train_config['use_scheduler'] :
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = train_config['lr_decay_rate'])
    else:
        lr_scheduler = None

    # Create a folder (if not exist already) to store temporary file during training
    os.makedirs(train_config['path_to_save_model'], exist_ok = True)

    # (OPTIONAL)
    if train_config['wandb_training']: wandb.watch(model, log = "all", log_freq = train_config['log_freq'])
    
    # Saved the current best loss on validation set
    best_loss_val = float('inf')

    # Moved model to device
    model.to(device)

    for epoch in range(train_config['epochs']):
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # (MANDATORY) Advance epoch, check validation loss and save the network

        # Advance epoch for train set (backward pass) and validation (no backward pass)
        train_loss      = HydraNet.train_epoch(model, loss_function, optimizer, train_loader, train_config, log_dict)
        validation_loss = HydraNet.validation_epoch(model, loss_function, validation_loader, train_config, log_dict)
        
        # Save the new BEST model if a new minimum is reach for the validation loss
        if validation_loss < best_loss_val:
            best_loss_val = validation_loss
            torch.save(model.state_dict(), '{}/{}'.format(train_config['path_to_save_model'], 'model_BEST.pth'))

        # Save the model after the epoch
        # N.b. When the variable epoch is n the model is trained for n + 1 epochs when arrive at this instructions.
        if (epoch + 1) % train_config['epoch_to_save_model'] == 0:
            torch.save(model.state_dict(), '{}/{}'.format(train_config['path_to_save_model'], "model_{}.pth".format(epoch + 1)))

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # (OPTIONAL) Optional steps during the training

        # Save the metrics in the log
        if train_config['measure_metrics_during_training']:
            compute_and_save_accuracy(model, train_loader, train_config, log_dict, 'TRAIN')
            compute_and_save_accuracy(model, validation_loader, train_config, log_dict, 'VALIDATION')

        # (OPTIONAL) Update learning rate (if a scheduler is provided)
        if lr_scheduler is not None:
            # Save the current learning rate if I load the data on wandb
            if train_config['wandb_training']: log_dict['learning_rate'] = optimizer.param_groups[0]['lr']

            # Update scheduler
            lr_scheduler.step()

        # (OPTIONAL) Print loss
        if train_config['print_var']:
            print("Epoch:{}".format(epoch))
            print("\t Train loss        = {}".format(train_loss.detach().cpu().float()))
            print("\t Validation loss   = {}".format(validation_loss.detach().cpu().float()))

            if lr_scheduler is not None: print("\t Learning rate     = {}".format(optimizer.param_groups[0]['lr']))
            if train_config['measure_metrics_during_training']:
                for el in log_dict : 
                    if 'Accuracy' in el : print("\t {} = {}".format(el, log_dict[el]))

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Log data on wandb

        # Update the log with the epoch losses
        log_dict['train_loss'] = train_loss
        log_dict['validation_loss'] = validation_loss
    
        
        # Add the model to the artifact
        if (epoch + 1) % train_config['epoch_to_save_model'] == 0:
            add_file_to_artifact(model_artifact, '{}/{}'.format(train_config['path_to_save_model'], "model_{}.pth".format(epoch + 1)))
        
        wandb.log(log_dict)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        
        # End training cycle
    
    # Save the model with the best loss on validation set
    if train_config['wandb_training']:
        add_file_to_artifact(model_artifact, '{}/{}'.format(train_config['path_to_save_model'], 'model_BEST.pth'))
    
    # Log the model artifact
    run.log_artifact(model_artifact)
