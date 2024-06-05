"""
Implementation of various neural networks for classification with the merged dataset
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import torch
from torch import nn
import pprint
from sklearn.metrics import accuracy_score

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
                    nn.Softmax(dim = 1)
                )
            )

        # List for the type of data to use for each head
        self.heads_source = config_heads['head_source']
    
    def forward(self, x_mems_1, x_mems_2, source_array) :
        x_mems_1 = self.input_mems_1(x_mems_1)
        x_mems_2 = self.input_mems_2(x_mems_2)
        x = torch.cat([x_mems_1, x_mems_2], dim = 1).float()
        x = self.body(x)
        
        heads_output = []
        for i in range(len(self.heads_source)) :
            source = self.heads_source[i]
            idx_source = source_array == source
            x_source = x[idx_source]
            heads_output.append(self.heads[i](x_source))

        return heads_output

    def classify(self, x_mems_1, x_mems_2, source_array, return_as_index = True):
        """
        Directly classify an input by returning the label (return_as_index = True) or the probability distribution on the labels (return_as_index = False)
        """
        
        labels_per_head_list = self.forward(x_mems_1, x_mems_2, source_array)

        if return_as_index:
            for i in range(len(labels_per_head_list)):
                labels_per_head = labels_per_head_list[i]
                predict_prob = torch.squeeze(torch.exp(labels_per_head).detach())
                labels_per_head_list[i] = torch.argmax(predict_prob, dim = 1)

        return labels_per_head_list
    
    def compute_accuracy_batch(self, x_mems_1, x_mems_2, source_array, true_labels) :
        labels_per_head_list = self.classify(x_mems_1, x_mems_2, source_array, return_as_index = True)
        accuracy_per_head_list = []
        for i in range(len(labels_per_head_list)):
            predicted_labels_per_head = labels_per_head_list[i]
            true_labels_head = true_labels[source_array == self.heads_source[i]]
            accuracy_per_head_list[i] = accuracy_score(true_labels_head.cpu().numpy(), predicted_labels_per_head .cpu().numpy())

        return accuracy_per_head_list

def train_epoch(model, loss_function, optimizer, train_loader, train_config, log_dict = None):
    # Set the model in training mode
    model.train()

    # Variable to accumulate the loss
    train_loss = 0

    for sample_data_batch, sample_label_batch in train_loader:
        # Move data to training device
        x_mems_1, x_mems_2, true_labels, labels_text, source_array = sample_data_batch.to(train_config['device'])
        true_labels = sample_label_batch.to(train_config['device'])

        # Zeros past gradients
        optimizer.zero_grad()
        
        # Networks forward pass
        out = model(x_mems_1, x_mems_2, source_array)

        # Compute the loss for each head
        batch_train_loss = 0
        for i in range(len(model.heads_source)) :
            # Get the labels for each head
            predict_labels_head = out[i]
            true_labels_head = true_labels[source_array == model.heads_source[i]]

            # Loss evaluation
            batch_train_loss += loss_function(predict_labels_head, true_labels_head)
    
        # Backward/Optimization pass
        batch_train_loss.backward()
        optimizer.step()

        # Accumulate the loss
        train_loss += batch_train_loss * x_mems_1.shape[0]

    # Compute final loss
    train_loss = train_loss / len(train_loader.sampler)

    if log_dict is not None:
        log_dict['train_loss'] = float(train_loss)
        print("TRAIN LOSS")
        pprint.pprint(log_dict)
    
    return train_loss


def validation_epoch(model, loss_function, validation_loader, train_config, log_dict = None):
    # Set the model in training mode
    model.eval()

    # Variable to accumulate the loss
    validation_loss = 0

    with torch.no_grad():

        for sample_data_batch, sample_label_batch in validation_loader:
            # Move data to training device
            x_mems_1, x_mems_2, true_labels, labels_text, source_array = sample_data_batch.to(train_config['device'])
            true_labels = sample_label_batch.to(train_config['device'])

            # Networks forward pass
            out = model(x_mems_1, x_mems_2, source_array)

            # Compute the loss for each head
            batch_validation_loss  = 0
            for i in range(len(model.heads_source)) :
                # Get the labels for each head
                predict_labels_head = out[i]
                true_labels_head = true_labels[source_array == model.heads_source[i]]

                # Loss evaluation
                batch_validation_loss += loss_function(predict_labels_head, true_labels_head)

            # Accumulate the loss
            validation_loss += batch_validation_loss * x_mems_1.shape[0]

        # Compute final loss
        validation_loss  = validation_loss  / len(validation_loader.sampler)

    if log_dict is not None:
        log_dict['train_loss'] = float(validation_loss)
        print("TRAIN LOSS")
        pprint.pprint(log_dict)
    
    return validation_loss
