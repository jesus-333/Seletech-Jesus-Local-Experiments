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
path_save_beans = 'data/merged_dataset/beans.csv'

path_orange_calib = 'data/Orange/fruit_orange_calib.csv'
path_orange_calib = 'data/Orange/fruit_orange.csv'
path_save_orange = 'data/merged_dataset/orange.csv'

path_potos_spectra = 'data/[2021-08-05_to_11-26]All_PlantSpectra.csv'
path_potos_water_timestamp = 'data/[2021-08-05_to_11-26]PlantTest_Notes.csv'
time_interval_start = 45 # After this value in minutes the spectra are considered wet
time_interval_end = 360 # After this value in minutes the spectra are considered dry agai
path_save_potos = 'data/merged_dataset/potos.csv'

path_powder = 'data/dataset_whole_PMT.pkl'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Get data beans
print("Beans START")
data_beans = merge_data_utils.merge_beans_spectra(t_list_beans)
data_beans = merge_data_utils.create_new_dataframe_beans(data_beans, labels_type_beans, preprocess_config, n_spectra_per_source)
data_beans.to_csv(path_save_beans)
print("Beans END\n")

# Get data orange
print("Orange START")
data_orange = pd.read_csv(path_orange_calib)
data_orange = merge_data_utils.create_new_dataframe_orange(data_orange, preprocess_config, n_spectra_per_source)
data_orange.to_csv(path_save_orange)
print("Orange END\n")

# Get potos data
print("Potos START")
# data_potos_full, wavelength, timestamp = merge_data_utils.load_spectra_data_potos(path_potos_spectra, return_numpy = True)
# data_potos = pd.DataFrame(data_potos_full, columns = wavelength.astype(int).astype(str))
# data_potos = merge_data_utils.create_new_dataframe_potos(data_potos, timestamp, path_potos_water_timestamp, time_interval_start, time_interval_end, preprocess_config, n_spectra_per_source)
# data_potos.to_csv(path_save_potos)
print("Potos END")
