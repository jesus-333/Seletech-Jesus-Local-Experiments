"""
Plot all the spectra, for a specific day and lampo power, separated by group.
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

t = 1

mems_to_plot = 2

plot_config = dict(
    figsize = (16, 6),
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

# Get data for specific memes
if mems_to_plot == 1 :
    tmp_wave_1 = int(1350)
    tmp_wave_2 = int(1650)
elif mems_to_plot == 2 :
    tmp_wave_1 = int(1750)
    tmp_wave_2 = int(2150)
elif mems_to_plot == 'both' :
    tmp_wave_1 = 1350
    tmp_wave_2 = 2150
else:
    raise ValueError("mems_to_plot must have value 1 or 2 or both")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Load and preprocess data

# Load data
path_spectra = "data/beans/t{}/csv/beans_avg.csv".format(t)
# path_spectra = "data/beans/t{}/csv/beans.csv".format(t)
spectra_data_ALL, wavelength, group_labels_list, plant_labels_list, plant_type_list = manage_data_beans.read_data_beans_single_file(path_spectra, return_numpy = False)
spectra_data = spectra_data_ALL[spectra_data_ALL['gain_0'] == 1]
if plant is not None : spectra_data = spectra_data[spectra_data['plant'] == plant]
if lamp_power is not None : spectra_data = spectra_data[spectra_data['lamp_0'] == lamp_power]

# Convert data into numpy array and create x-axis array
tmp_wavelength = wavelength[np.logical_and(wavelength >= tmp_wave_1, wavelength <= tmp_wave_2)]

# Plot the data
fig, axs = plt.subplots(1, len(group_list), figsize = plot_config['figsize'])

for i in range(len(group_list)) :
    ax = axs[i]
    group = group_list[i]
    
    tmp_spectra = spectra_data[spectra_data['test_control'] == group]
    tmp_spectra = tmp_spectra.loc[:, str(tmp_wave_1):str(tmp_wave_2)].to_numpy()
    ax.plot(tmp_wavelength, tmp_spectra.T, 
            color = group_color[group], linestyle = group_linestyle[group]
            )

    ax.legend()
    ax.grid()
    ax.set_xlabel('Wavelength [nm]')
    ax.set_ylabel('Intensity')
    ax.set_title('Group: {}'.format(group))

fig.tight_layout()
fig.show()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Compute difference in amplitude between different wavelengths given two plants

path_spectra = "data/beans/t{}/csv/beans_avg.csv".format(t)
# path_spectra = "data/beans/t{}/csv/beans.csv".format(t)
spectra_data_ALL, wavelength, group_labels_list, plant_labels_list, plant_type_list = manage_data_beans.read_data_beans_single_file(path_spectra, return_numpy = False)
spectra_data = spectra_data_ALL[spectra_data_ALL['gain_0'] == 1]
if plant is not None : spectra_data = spectra_data[spectra_data['plant'] == plant]
if lamp_power is not None : spectra_data = spectra_data[spectra_data['lamp_0'] == lamp_power]

fig, axs = plt.subplots(1, 3, figsize = plot_config['figsize'])
for i in range(len(group_list)) :
    group = group_list[i]
    ax = axs[i]

    if group == 'control' :
        plant_list = ['CON1', 'CON2']
    elif group == 'test_150' :
        plant_list = ['NACL150_1', 'NACL150_2']
    elif group == 'test_300' :
        plant_list = ['NACL300_1', 'NACL300_2']
    
    # Get the data for the group
    tmp_spectra = spectra_data[spectra_data['test_control'] == group]

    # Get the spectra for the two plants
    spectra_1 = tmp_spectra[tmp_spectra['type'] == plant_list[0]].loc[:, str(tmp_wave_1):str(tmp_wave_2)].to_numpy()
    spectra_2 = tmp_spectra[tmp_spectra['type'] == plant_list[1]].loc[:, str(tmp_wave_1):str(tmp_wave_2)].to_numpy()

    # Compute the difference in amplitude
    diff_spec = np.abs(spectra_1 - spectra_2).squeeze()

    ax.plot(tmp_wavelength, diff_spec)
    ax.grid()
    ax.set_xlabel('Wavelength [nm]')
    ax.set_title('{}'.format(group))

    print("The average difference is {:.2f}±{:.2f}".format(np.mean(diff_spec), np.std(diff_spec)))

fig.suptitle('Difference between Plant {} and Plant {}'.format(plant_list[0][-1], plant_list[1][-1]))
fig.tight_layout()
fig.show()


