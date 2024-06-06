"""
Implementation of various neural networks for classification with the merged dataset
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import torch
from torch import nn
import numpy as np
import pprint
from sklearn.metrics import accuracy_score

from . import metrics

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

class hydra_net_v1(nn.Module) :
    def __init__(self, config_body : dict, config_heads : dict) :
        """
        Create a HydraNet model with n heads and a common body.
        """
        super().__init__()

        activation = nn.ELU()
        
        # Input layer for the two mems
        self.input_mems_1 = nn.Sequential(
            nn.Linear(config_body['input_size_mems_1'], int(config_body['hidden_size'][0] / 2)),
            activation
        )
        self.input_mems_2 = nn.Sequential(
            nn.Linear(config_body['input_size_mems_2'], int(config_body['hidden_size'][0] / 2)),
            activation
        )
        
        # Create common body
        self.body = nn.ModuleList()
        for i in range(1, len(config_body['hidden_size'])) :
            input_neurons = config_body['hidden_size'][i - 1]
            output_neurons = config_body['hidden_size'][i]
            tmp_layer = nn.Linear(input_neurons, output_neurons)
            self.body.append(tmp_layer)
            self.body.append(activation)
        self.body = nn.Sequential(*self.body)
        
        # Create heads
        self.heads = nn.ModuleList()
        for i in range(config_heads['n_heads']) :
            self.heads.append(
                nn.Sequential(
                    nn.Linear(config_body['hidden_size'][-1], config_heads['output_size'][i]),
                    nn.LogSoftmax(dim = 1)
                )
            )

        # List for the type of data to use for each head
        self.head_sources = config_heads['head_sources']
    
    def forward(self, x_mems_1, x_mems_2, source_array) :
        x_mems_1 = self.input_mems_1(x_mems_1)
        x_mems_2 = self.input_mems_2(x_mems_2)
        x = torch.cat([x_mems_1, x_mems_2], dim = 1).float()
        x = self.body(x)
        
        heads_output_list = []
        for i in range(len(self.head_sources)) :
            source = self.head_sources[i]
            idx_source = np.asarray(source_array) == source
            x_source = x[idx_source]
            heads_output = self.heads[i](x_source)
            heads_output_list.append(heads_output)

        return heads_output_list

    def classify(self, x_mems_1, x_mems_2, source_array, return_as_index = True):
        """
        Directly classify an input by returning the label (return_as_index = True) or the probability distribution on the labels (return_as_index = False)
        """
        
        with torch.no_grad() :
        
            labels_per_head_list = self.forward(x_mems_1, x_mems_2, source_array)
    
            if return_as_index:
                for i in range(len(labels_per_head_list)):
                    labels_per_head = labels_per_head_list[i]
                    predict_prob = torch.squeeze(torch.exp(labels_per_head).detach())
                    labels_per_head_list[i] = torch.argmax(predict_prob, dim = 1)
    
            return labels_per_head_list
    
    def compute_metrics_batch(self, x_mems_1, x_mems_2, source_array, true_labels) :
        """
        Compute accuracy, cohen_kappa, sensitivity, specificity, f1 and confusion matrix for each head
        """
        labels_per_head_list = self.classify(x_mems_1, x_mems_2, source_array, return_as_index = True)
        metrics_per_head_list = []
        for i in range(len(labels_per_head_list)):
            predicted_labels_head = labels_per_head_list[i]
            true_labels_head = true_labels[source_array == self.head_sources[i]]
            metrics_per_head_list.append(metrics.compute_metrics_from_labels(true_labels_head.cpu(), predicted_labels_head.cpu()))

        return metrics_per_head_list

def train_epoch(model, loss_function, optimizer, train_loader, train_config, log_dict = None):
    # Set the model in training mode
    model.train()

    # Variables to accumulate the loss
    train_loss = 0
    loss_per_head = {}
    for head_sources in model.head_sources : loss_per_head[head_sources] = 0

    for x_mems_1, x_mems_2, true_labels, labels_text, source_array in train_loader:
        # Move data to training device
        x_mems_1 = x_mems_1.to(train_config['device'])
        x_mems_2 = x_mems_2.to(train_config['device'])
        true_labels = true_labels.to(train_config['device'])
        source_array = np.asarray(source_array)

        # Zeros past gradients
        optimizer.zero_grad()
        
        # Networks forward pass
        out = model(x_mems_1, x_mems_2, source_array)

        # Compute the loss for each head
        batch_train_loss = 0
        for i in range(len(model.head_sources)) :
            # Get the labels for each head
            predict_labels_head = out[i]
            true_labels_head = true_labels[source_array == model.head_sources[i]]

            # Loss evaluation
            head_loss = loss_function(predict_labels_head, true_labels_head)
            batch_train_loss += head_loss
            loss_per_head[model.head_sources[i]] = head_loss
    
        # Backward/Optimization pass
        batch_train_loss.backward()
        optimizer.step()

        # Accumulate the loss
        train_loss += batch_train_loss * x_mems_1.shape[0]

    # Compute final loss
    train_loss = train_loss / len(train_loader.sampler)

    if log_dict is not None:
        log_dict['train_loss_total'] = float(train_loss)
        for i in range(len(model.head_sources)) : log_dict['train_loss_{}'.format(model.head_sources[i])] = float(loss_per_head[model.head_sources[i]])
        
        # print("TRAIN LOSS")
        # pprint.pprint(log_dict)
        
    return train_loss


def validation_epoch(model, loss_function, validation_loader, train_config, log_dict = None):
    # Set the model in training mode
    model.eval()

    # Variable to accumulate the loss
    validation_loss = 0
    loss_per_head = {}
    for head_sources in model.head_sources : loss_per_head[head_sources] = 0

    with torch.no_grad():

        for x_mems_1, x_mems_2, true_labels, labels_text, source_array in validation_loader:
            # Move data to training device
            x_mems_1 = x_mems_1.to(train_config['device'])
            x_mems_2 = x_mems_2.to(train_config['device'])
            true_labels = true_labels.to(train_config['device'])
            source_array = np.asarray(source_array)

            # Networks forward pass
            out = model(x_mems_1, x_mems_2, source_array)

            # Compute the loss for each head
            batch_validation_loss  = 0
            for i in range(len(model.head_sources)) :
                # Get the labels for each head
                predict_labels_head = out[i]
                true_labels_head = true_labels[source_array == model.head_sources[i]]

                # Loss evaluation
                head_loss = loss_function(predict_labels_head, true_labels_head)
                batch_validation_loss += head_loss
                loss_per_head[model.head_sources[i]] = head_loss

            # Accumulate the loss
            validation_loss += batch_validation_loss * x_mems_1.shape[0]

        # Compute final loss
        validation_loss  = validation_loss  / len(validation_loader.sampler)

    if log_dict is not None:
        log_dict['validation_loss_total'] = float(validation_loss)
        for i in range(len(model.head_sources)) : log_dict['validation_loss_{}'.format(model.head_sources[i])] = float(loss_per_head[model.head_sources[i]])
        
        # print("TRAIN LOSS")
        # pprint.pprint(log_dict)
    
    return validation_loss
