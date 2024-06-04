"""
Merge the data from the different sources in a single dataset
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import pandas as pd

from library import merge_data_utils

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings

n_spectra_per_source = 800

preprocess_config = dict(
    compute_absorbance = True,
    use_SNV = True,
    use_sg_filter = True,
    w = 30,
    p = 3,
    deriv = 2,
    use_minmax = True,
)

t_list_beans = [0, 1, 2]
labels_type_beans = 1

path_orange_calib = 'data/Orange/fruit_orange_calib.csv'
path_potos = 'data/jesus_spectra_full_dataset.csv'
path_powder = 'data/dataset_whole_PMT.pkl'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Get data beans
data_beans = merge_data_utils.merge_beans_spectra(t_list_beans)
data_beans = merge_data_utils.create_new_dataframe_beans(data_beans, labels_type_beans, preprocess_config, n_spectra_per_source)

# Get data orange
data_orange = pd.read_csv(path_orange_calib)
data_orange = merge_data_utils.create_new_dataframe_orange(data_orange, labels_type_beans, preprocess_config, n_spectra_per_source)

data_potos = pd.read_csv(path_potos)
