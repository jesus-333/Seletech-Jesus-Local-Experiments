"""
Implementation of various neural networks for classification with the merged dataset
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import torch
from torch import nn

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

