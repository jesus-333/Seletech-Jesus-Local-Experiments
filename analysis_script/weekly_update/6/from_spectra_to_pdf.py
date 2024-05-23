"""
Convert the curve of the spectra in probability density functions
TODO INCOMPLETE
"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

from library import manage_data_beans, preprocess

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

plant = 'PhaseolusVulgaris'
plant = 'ViciaFaba'

lamp_power = 80

t_list = [0, 5]

# Parameter for preprocess
use_SNV = False
compute_absorbance = True
use_sg_filter = True
w = 30
p = 3
deriv = 2

mems_to_plot = 1

plot_config = dict(
    figsize = (12, 8),
    fontsize = 20,
    # ylim = [2150, 2950],
    save_fig = True,
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Other variable used during the scripts

group_color = {'test_300' : 'red', 'control' : 'green', 'test_150' : 'blue'}
group_linestyle = {'test_300' : 'dashed', 'control' : 'solid', 'test_150' : 'dashdot'}
group_linewidth = {'test_300' : 4, 'control' : 1, 'test_150' : 2}

if plant == 'ViciaFaba' :
    group_list = ['test_300', 'test_150', 'control']
else :
    group_list = ['test_150', 'control']

plt.rcParams.update({'font.size': plot_config['fontsize']})

linestyle_list = ['solid', 'dashed']
color_list = ['green', 'red']

# Set the window to zero if the sg filter is not used. The window is used also to cut the border of the signal in the script
if not use_sg_filter : w = 0

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Load and preprocess data

# Load data
path_spectra = "data/beans/t{}/csv/beans_avg.csv".format(t_to_compare)
spectra_data_ALL, wavelength, group_labels_list, plant_labels_list, plant_type_list = manage_data_beans.read_data_beans_single_file(path_spectra, return_numpy = False)
spectra_data = spectra_data_ALL[spectra_data_ALL['gain_0'] == 1]
if plant is not None : spectra_data = spectra_data[spectra_data['plant'] == plant]
if lamp_power is not None : spectra_data = spectra_data[spectra_data['lamp_0'] == lamp_power]
