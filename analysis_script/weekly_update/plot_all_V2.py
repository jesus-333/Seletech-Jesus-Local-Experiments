"""
Remove all spectra with amplitude below 1000 and color them according to plant group (control, test 150 or test 300)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from library import manage_data_beans, preprocess

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

threshold = 1000

# plant = 'PhaseolusVulgaris'
plant = 'ViciaFaba'
plant = None

remove_mean = False

plot_config = dict(
    figsize = (20, 10),
    fontsize = 15,
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

plt.rcParams.update({'font.size': plot_config['fontsize']})

def split_data_per_mems(data_dataframe, remove_mean = False):
    data_mems_1 = data_dataframe.loc[:, "1350":"1650"].to_numpy().squeeze()
    data_mems_2 = data_dataframe.loc[:, "1750":"2150"].to_numpy().squeeze()

    if remove_mean : 
        if len(data_mems_1.shape) > 1:
            data_mems_1 = ( data_mems_1.T - data_mems_1.mean(1) ).T
            data_mems_2 = ( data_mems_2.T - data_mems_2.mean(1) ).T
        else : 
            data_mems_1 = data_mems_1 - data_mems_1.mean()
            data_mems_2 = data_mems_2 - data_mems_2.mean()

    return data_mems_1, data_mems_2

def filter_spectra_by_threshold(spectra_dataframe, threshold : int):
    spectra_data = spectra_dataframe.loc[:, "1350":"2150"].to_numpy().squeeze()

    # idx_data_to_remove = ((spectra_data < threshold).sum(1) >= 75)
    idx_data_to_keep = ((spectra_data > threshold).sum(1) >= 600) # Tieni lo spettro solo se 650 delle 702 lunghezze d'onda registrate sono superiori alla threshold
    
    return spectra_dataframe[idx_data_to_keep]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

t_list = [0, 1, 2, 3, 4, 5]
plant_group_list = ['control', 'test_150', 'test_300']
group_to_color = {'control' : 'black', 'test_150' : 'green', 'test_300' : 'orange'}

wavelengts_1 = np.arange(1350, 1650 + 1)
wavelengts_2 = np.arange(1750, 2150 + 1)

gain = 1

fig_1, axs_1 = plt.subplots(2, 3, figsize = plot_config['figsize'])
fig_2, axs_2 = plt.subplots(2, 3, figsize = plot_config['figsize'])

for i in range(2):
    for j in range(3):
        t = t_list[i * 3 + j]

        path_spectra = "data/beans/t{}/csv/beans_avg.csv".format(t)
        spectra_data, wavelength, group_labels_list, plant_labels_list, plant_type_list = manage_data_beans.read_data_beans_single_file(path_spectra, return_numpy = False)
        spectra_data = filter_spectra_by_threshold(spectra_data, threshold)
        if plant is not None : spectra_data = spectra_data[spectra_data['plant'] == plant]

        for k in range(len(plant_group_list)):
            plant_group = plant_group_list[k]

            # Filter by lamp power and gain (Spectra data)
            tmp_spectra_data = spectra_data[spectra_data['gain_0'] == gain]
            tmp_spectra_data = tmp_spectra_data[tmp_spectra_data['test_control'] == plant_group]

            # Get xtalk, srs, wcs and spectra data in separate variables for each mems 
            spectra_data_1, spectra_data_2 = split_data_per_mems(tmp_spectra_data, remove_mean)

            # Plot mems 1
            axs_1[i, j].plot(wavelengts_1, spectra_data_1.T, color = group_to_color[plant_group])

            # Plot mems 2
            axs_2[i, j].plot(wavelengts_2, spectra_data_2.T, color = group_to_color[plant_group])

        axs_1[i, j].set_title("Spectra t{}".format(t))
        axs_2[i, j].set_title("Spectra t{}".format(t))

for i in range(2):
    for j in range(3):
        axs_1[i, j].set_xlabel('Wavelength [nm]')
        axs_1[i, j].grid(True)

        axs_2[i, j].set_xlabel('Wavelength [nm]')
        axs_2[i, j].grid(True)

axs_2[i, j].legend(group_to_color)
axs_1[i, j].legend(group_to_color)

leg_1 = axs_1[i, j].get_legend()
leg_2 = axs_2[i, j].get_legend()
color_list = ['black', 'green', 'orange']
for i in range(3):
    leg_1.legendHandles[i].set_color(color_list[i])
    leg_2.legendHandles[i].set_color(color_list[i])

fig_1.tight_layout()
fig_1.show()

fig_2.tight_layout()
fig_2.show()
