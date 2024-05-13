"""
Compare the spectra at t0 between different groups to search differences caused by the instruments
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

from library import manage_data_beans, preprocess

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

plant = 'PhaseolusVulgaris'
# plant = 'ViciaFaba'

lamp_power = 80

# Parameter for preprocess
compute_absorbance = True
use_SNV = True
use_sg_filter = True
w = 30
p = 3
deriv = 2

mems_to_plot = 1

plot_config = dict(
    figsize = (20, 12),
    fontsize = 20,
    split_plot = False,
    save_fig = True,
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

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

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Load and preprocess data

# Load data
path_spectra = "data/beans/t0/csv/beans_avg.csv"
spectra_data_ALL, wavelength, group_labels_list, plant_labels_list, plant_type_list = manage_data_beans.read_data_beans_single_file(path_spectra, return_numpy = False)
spectra_data = spectra_data_ALL[spectra_data_ALL['gain_0'] == 1]
if plant is not None : spectra_data = spectra_data[spectra_data['plant'] == plant]
if lamp_power is not None : spectra_data = spectra_data[spectra_data['lamp_0'] == lamp_power]

# Remove spectra with at least a portion of spectra below the threshold
# spectra_data, idx_data_to_keep = preprocess.filter_spectra_by_threshold(spectra_data, threshold = min_amplitude, percentage_above_threshold = percentage_above_threshold)

# Preprocess
meta_data = spectra_data.loc[:, "timestamp":"type"]
if compute_absorbance : spectra_data = preprocess.R_A(spectra_data, keep_meta = False)
if use_SNV : spectra_data = preprocess.normalize_standardization(spectra_data, divide_mems = True)
if use_sg_filter : spectra_data = preprocess.sg(spectra_data, w, p, deriv, keep_meta = False)
if compute_absorbance or use_sg_filter: # Since during preprocess the metadata are removed here are restored
    spectra_data = pd.concat([meta_data, spectra_data], axis = 1)
else:
    raise ValueError("At least 1 between compute_absorbance and use_sg_filter must be true")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Plot data

for group in group_list :
    tmp_spectra = spectra_data[spectra_data['test_control'] == group]


