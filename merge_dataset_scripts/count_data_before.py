"""
Count the data from each source before extracting a subsample from each source for the merging

"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import pandas as pd
from library import merge_data_utils

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings

t_list_beans = [0, 1, 2]
path_orange_calib = 'data/Orange/fruit_orange_calib.csv'
path_potos = 'data/jesus_spectra_full_dataset.csv'
path_powder = 'data/dataset_whole_PMT.pkl'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Read data

data_beans = merge_data_utils.merge_beans_spectra(t_list_beans)

data_orange = pd.read_csv(path_orange_calib)

data_potos = pd.read_csv(path_potos)

data_poweder = pd.read_pickle(path_powder)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Show the count for each source
print('Beans  :', data_beans.shape, '(t_list = {})'.format(t_list_beans))
print('Orange :', data_orange.shape)
print('Potos  :', data_potos.shape)
print('Powder :', data_poweder.shape)


"""
Results of the script :

Results of the script:
Beans  : (819, 713) (t_list = [0, 1, 2])
Orange : (1920, 713)
Potos  : (158439, 709)
Powder : (517176, 709)
"""
