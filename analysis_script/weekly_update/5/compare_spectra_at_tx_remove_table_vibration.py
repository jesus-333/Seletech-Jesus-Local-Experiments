"""
Compare the spectra at t0 between different groups to search differences caused by the instruments
Remove from the average of the spectra the measurements acquired while the table was vibrating
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import json
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

from library import manage_data_beans, preprocess

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

plant = 'PhaseolusVulgaris'
plant = 'ViciaFaba'

lamp_power = 80

t_to_compare = 1

# Notes
# For ViciaFaba NACL300_1 and NACL300_5 are the plants that contribute the most to the amplitude in the std

# Parameter for preprocess
compute_absorbance = False
use_SNV = False
use_sg_filter = False
w = 30
p = 3
deriv = 2

mems_to_plot = 1

plot_config = dict(
    figsize = (12, 8),
    fontsize = 20,
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

tmp_path = 'analysis_script/weekly_update/5/plants_to_remove_for_vibrations.json'
with open(tmp_path, 'r') as j:
    plants_to_remove = json.loads(j.read())

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Load and preprocess data

# Load data
path_spectra = "data/beans/t{}/csv/beans_avg.csv".format(t_to_compare)
spectra_data_ALL, wavelength, group_labels_list, plant_labels_list, plant_type_list = manage_data_beans.read_data_beans_single_file(path_spectra, return_numpy = False)
spectra_data = spectra_data_ALL[spectra_data_ALL['gain_0'] == 1]

if plant is not None : spectra_data = spectra_data[spectra_data['plant'] == plant]
if lamp_power is not None : spectra_data = spectra_data[spectra_data['lamp_0'] == lamp_power]

# Remove plants with vibration
if str(t_to_compare) in plants_to_remove :
    if plant in plants_to_remove[str(t_to_compare)] :
        for plants_id in plants_to_remove[str(t_to_compare)][plant] :
            spectra_data = spectra_data[spectra_data['type'] != plants_id]
        
        group_list = list(set(spectra_data['test_control']))

# Remove spectra with at least a portion of spectra below the threshold
# spectra_data, idx_data_to_keep = preprocess.filter_spectra_by_threshold(spectra_data, threshold = min_amplitude, percentage_above_threshold = percentage_above_threshold)

# Preprocess
meta_data = spectra_data.loc[:, "timestamp":"type"]
if compute_absorbance : spectra_data = preprocess.R_A(spectra_data, keep_meta = False)
if use_SNV : spectra_data = preprocess.normalize_standardization(spectra_data, divide_mems = True)
if use_sg_filter : spectra_data = preprocess.sg(spectra_data, w, p, deriv, keep_meta = False)
if compute_absorbance or use_sg_filter: # Since during preprocess the metadata are removed here are restored
    spectra_data = pd.concat([meta_data, spectra_data], axis = 1)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Plot data

fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])

for group in group_list :
    tmp_spectra = spectra_data[spectra_data['test_control'] == group]
    
    # Get data for specific memes
    if mems_to_plot == 1 :
        tmp_wave_1 = int(1350 + w / 2)
        tmp_wave_2 = int(1650 - w / 2)
    elif mems_to_plot == 2 :
        tmp_wave_1 = int(1750 + w / 2)
        tmp_wave_2 = int(2150 - w / 2)
    elif mems_to_plot == 'both' :
        tmp_wave_1 = 1350
        tmp_wave_2 = 2150
    else:
        raise ValueError("mems_to_plot must have value 1 or 2 or both")
    
    # Convert data into numpy array and create x-axis array
    tmp_spectra = tmp_spectra.loc[:, str(tmp_wave_1):str(tmp_wave_2)].to_numpy()
    tmp_wavelength = wavelength[np.logical_and(wavelength >= tmp_wave_1, wavelength <= tmp_wave_2)]
    
    # Get average and std
    avg_spectra = tmp_spectra.mean(0)
    std_spectra = tmp_spectra.std(0)

    ax.plot(tmp_wavelength, avg_spectra, label = group, color = group_color[group])
    ax.fill_between(tmp_wavelength, avg_spectra - std_spectra, avg_spectra + std_spectra, color = group_color[group], alpha = 0.5)

ax.grid(True)
ax.legend()
ax.set_xlabel("Wavelength [nm]")
ax.set_xlim([tmp_wavelength[1], tmp_wavelength[-1]])

title = '{} - lamp {}'.format(plant, lamp_power)
if use_sg_filter : title += ' - w {} - p {} - der {}'.format(w, p, deriv)
ax.set_title(title)

fig.show()
fig.tight_layout()

if plot_config['save_fig'] :
    path_save = 'Saved Results/weekly_update_beans/5/compare_spectra_at_t0_without_vibration/'
    os.makedirs(path_save, exist_ok = True)

    path_save += 't_{}_{}_lamp_{}_mems_{}'.format(t_to_compare, plant, lamp_power, mems_to_plot)
    if compute_absorbance : path_save += '_absorbance'
    if use_SNV : path_save += '_SNV'
    if use_sg_filter : path_save += '_w_{}_p_{}_der_{}'.format(w, p, deriv)
    fig.savefig(path_save + ".png", format = 'png')
