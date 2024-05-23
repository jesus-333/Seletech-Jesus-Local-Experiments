"""
Check for each gruoup the contribution of each plant to the std of each day
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

wavelength_to_return = 1500

# Notes
# (t = 1) For ViciaFaba NACL300_1 and NACL300_5 are the plants that contribute the most to the amplitude in the std

# Parameter for preprocess
compute_absorbance = False
use_SNV = True
use_sg_filter = False
w = 30
p = 3
deriv = 2

mems_to_plot = 2

plot_config = dict(
    figsize = (12, 8),
    fontsize = 20,
    # ylim = [1450, 2300],
    ylim = [2000, 2800],
    save_fig = True,
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Other variable used during the scripts

def get_std_spectra(spectra_df, wavelength_array, wavelength_to_return) :
    """
    Compute the std for each wavelength and return the ones specified by wavelength_to_return
    """
    tmp_spectra = spectra_df.loc[:, str(wavelength_array[0]):str(wavelength_array[-1])].to_numpy()
    std_spectra = tmp_spectra.std(0)
    return std_spectra[wavelength_array == wavelength_to_return]

t_list = [0, 1, 2, 3, 4, 5]

group_color = {'test_300' : 'red', 'control' : 'green', 'test_150' : 'blue'}
group_linestyle = {'test_300' : 'dashed', 'control' : 'solid', 'test_150' : 'dashdot'}
group_linewidth = {'test_300' : 4, 'control' : 1, 'test_150' : 2}

if plant == 'ViciaFaba' :
    group_list = ['test_300', 'test_150', 'control']
else :
    group_list = ['test_150', 'control']

# Set the window to zero if the sg filter is not used. The window is used also to cut the border of the signal in the script
if not use_sg_filter : w = 0

tmp_path = 'analysis_script/weekly_update/5/plants_to_remove_for_vibrations.json'
with open(tmp_path, 'r') as j:
    plants_to_remove = json.loads(j.read())

# Matrix to save information about std
stats_std = np.zeros((len(group_list), len(t_list), 6))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

for i in range(len(t_list)) :

    t = t_list[i]

    # Load data
    path_spectra = "data/beans/t{}/csv/beans_avg.csv".format(t)
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

    for j in range(len(group_list)) : # Iterate groups
        
        # Get data of a specific group
        group = group_list[j]
        tmp_spectra_group = spectra_data[spectra_data['test_control'] == group]
        
        # Get plants for the specific group
        plants_list = list(set(tmp_spectra_group['type']))
        plants_list.sort()
        
        # Compute std with all spectra of all the plants
        stats_std[j, i, -1] = get_std_spectra(tmp_spectra_group, wavelength, wavelength_to_return)

        for k in range(len(plants_list)) : # Iterate through plants
            plants_id = plants_list[k]
            
            # Get all the plants different from plants_id
            tmp_sepctra_with_removed_plant = tmp_spectra_group[tmp_spectra_group['type'] != plants_id]

            # Compute std
            stats_std[j, i, k] = get_std_spectra(tmp_sepctra_with_removed_plant, wavelength, wavelength_to_return)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

group_idx = 0
stats_std_group = stats_std[group_idx]

xticks_labels = []
for i in range(stats_std_group.shape[1]):
    stats_std_group[:, i] = stats_std_group[:, i] / stats_std_group[:, -1]
    xticks_labels.append("Pl. {}".format(i + 1))
xticks_labels[-1] = 'ALL'

fig, ax = plt.subplots(1, 1)
tmp_plot = ax.imshow(stats_std_group)
fig.colorbar(tmp_plot)

xticks = np.asarray([0, 1, 2, 3, 4, 5])
ax.set_xticks(xticks, labels = xticks_labels, fontsize = plot_config['fontsize'])
ax.set_title("Std for {}".format(group_list[group_idx]))

plt.show()
