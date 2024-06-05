"""
Contains classes and functions to handle NIRS data
"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import torch
import os
import pandas as pd
import numpy as np

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class NIRS_dataset_merged(torch.utils.data.Dataset):
    def __init__(self, source_path_list : list, idx_to_ketp = None):
        
        # Get data and merge them into a single dataframe
        merge_dataframe = pd.DataFrame()
        for path in source_path_list:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File {path} does not exist")
            else :
                tmp_dataframe = pd.read_csv(path)
                merge_dataframe = pd.concat([merge_dataframe, tmp_dataframe], axis = 0)
                
        if idx_to_ketp is not None : merge_dataframe = merge_dataframe.iloc[idx_to_ketp, :]
        
        # Get data for the two MEMS
        wave_1_mems_1, wave_2_mems_1, wave_1_mems_2, wave_2_mems_2 = self.compute_parameter_preprocess(merge_dataframe)
        self.data_mems_1 = merge_dataframe.loc[:, wave_1_mems_1:wave_2_mems_1].values.astype(float)
        self.data_mems_2 = merge_dataframe.loc[:, wave_1_mems_2:wave_2_mems_2].values.astype(float)

        # Get labels (both numerical and textual)
        self.label = merge_dataframe.loc[:, 'label'].values.astype(int)
        self.label_text = merge_dataframe.loc[:, 'label_text'].values.astype(str)
        self.source = merge_dataframe.loc[:, 'source'].values.astype(str)

        # Compute wavelengths array
        self.wavelengths_1 = np.arange(int(wave_1_mems_1), int(wave_2_mems_1) + 1)
        self.wavelengths_2 = np.arange(int(wave_1_mems_2), int(wave_2_mems_2) + 1)

        self.convert_to_torch_tensor()

    def __len__(self):
        return self.data_mems_1.shape[0]

    def __getitem__(self, idx):
        return self.data_mems_1[idx], self.data_mems_2[idx], self.label[idx], self.label_text[idx], self.source[idx]

    def compute_parameter_preprocess(self, merge_dataframe : pd.DataFrame, return_as_string : bool = True) :
        starting_wavelength_dataframe = int(merge_dataframe.columns[1])
        w = (starting_wavelength_dataframe - 1350) * 2
        tmp_wave_1_mems_1 = int(1350 + w / 2)
        tmp_wave_2_mems_1 = int(1650 - w / 2)
        tmp_wave_1_mems_2 = int(1750 + w / 2)
        tmp_wave_2_mems_2 = int(2150 - w / 2)

        if return_as_string :
            return str(tmp_wave_1_mems_1), str(tmp_wave_2_mems_1), str(tmp_wave_1_mems_2), str(tmp_wave_2_mems_2)
        else :
            return tmp_wave_1_mems_1, tmp_wave_2_mems_1, tmp_wave_1_mems_2, tmp_wave_2_mems_2
    
    def convert_to_torch_tensor(self) :
        self.data_mems_1 = torch.from_numpy(self.data_mems_1).float()
        self.data_mems_2 = torch.from_numpy(self.data_mems_2).float()
        self.label = torch.from_numpy(self.label).long()


def get_idx_to_split_data(n_elements : int, percentage_split : float, seed = -1):
    """
    Get to list of indices to split an array of data.
    """
    # Use of the seed for reproducibility
    np.random.seed(seed)

    # Create idx vector
    idx = np.random.permutation(n_elements)
    size_1 = int(n_elements * percentage_split)
    
    return idx[0:size_1], idx[size_1:]
