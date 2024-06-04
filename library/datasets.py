"""
Contains classes and functions to handle NIRS data
"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import torch

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class NIRS_dataset(torch.utils.data.Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
