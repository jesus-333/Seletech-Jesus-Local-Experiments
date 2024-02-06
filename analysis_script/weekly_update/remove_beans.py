import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from library import manage_data_beans, preprocess

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

t_list = [0, 1, 2, 3, 4, 5]

gain = 1

# plant = 'PhaseolusVulgaris'
plant = 'ViciaFaba'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

for i in range(len(t_list)):
    t = t_list[i]

    path_spectra = "data/beans/t{}/csv/beans_avg.csv".format(t)
    spectra_data, wavelength, group_labels_list, plant_labels_list, plant_type_list = manage_data_beans.read_data_beans_single_file(path_spectra, return_numpy = False)
