"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Support functions related to the datasets

"""

import numpy as np
import pandas as pd

import torch
from torch import nn


#%% Dataset declaration

class PytorchDatasetPlantSpectra_V1(torch.utils.data.Dataset):
    """
    Extension of PyTorch Dataset class to work with spectra plants data.
    """
    
    # Inizialization method
    def __init__(self, filename, normalize_trials):
        spectra_data, wavelength, timestamp = self.load_spectra_data(filename, normalize_trials)
        self.spectra = torch.from_numpy(spectra_data).float() 
      
    def __getitem__(self, idx):
        return self.spectra[idx, :]
    
    def __len__(self):
        return self.spectra.shape[0]



    def load_spectra_data(self, filename, normalization_type = -1, print_var = True):
        spectra_plants_df = pd.read_csv(filename, header = 2, encoding= 'unicode_escape')
        
        # Convert spectra dataframe in a numpy matrix
        spectra_plants_numpy = spectra_plants_df.iloc[:,5:-1].to_numpy(dtype = float)
        if(print_var): print("Spectra matrix shape\t=\t", spectra_plants_numpy.shape, "\t (Time x Wavelength)")
        
        # Recover wavelength
        wavelength = spectra_plants_df.keys()[5:-1].to_numpy(dtype = float)
        if(print_var): print("Wavelength vector shape\t=\t", wavelength.shape)
        
        # Recover timestamp
        timestamp = spectra_plants_df['# Timestamp'].to_numpy()
        if(print_var): print("Timestamp vector shape\t=\t", timestamp.shape)
        
        if(normalization_type == 0 or normalization_type == 1): 
            spectra_plants_numpy = self.spectra_normalization(spectra_plants_numpy, normalization_type)
        
        return spectra_plants_numpy, wavelength, timestamp
    
    def compute_normalization_factor(self, spectra, norm_type):
        if(norm_type == 0): # Half normalization
            tmp_sum = np.sum(spectra, 0)
            normalization_factor = tmp_sum / spectra.shape[0]
        elif(norm_type == 1): # Full normalization
            tmp_sum = np.sum(spectra)
            normalization_factor = tmp_sum / (spectra.shape[0] * spectra.shape[1])
        else: 
            normalization_factor = 0

        return normalization_factor


    def spectra_normalization(self, spectra, norm_type):
        return spectra / self.compute_normalization_factor(spectra, norm_type)
        