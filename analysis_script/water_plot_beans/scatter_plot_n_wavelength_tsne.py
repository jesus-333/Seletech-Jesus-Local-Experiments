"""
Select n wavelength, reduce dimension with TSNE and plot them through scatter plot
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE

from library import manage_data_other_sensors, manage_data_beans, timestamp_functions
from library import preprocess

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Settings 

plant_to_examine = 'PhaseolusVulgaris'
plant_to_examine = 'ViciaFaba'

use_standardization = True
use_sg_preprocess = False

t_list = [0, 1, 2, 3, 4, 5, 6]
# t_list = [0]

wavelength_list = [1350, 1450, 1950, 2054, 2075, 1768, 1818, 1850]

plot_config = dict(
    figsize = (12, 8),
    fontsize = 15,
    save_fig = True
)

color_type = 1

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%%  

# Group of plants (control = no salt, test_150 = 150ml of salt, test_300 = 300ml of salt)
plant_group_list = ['control', 'test_150', 'test_300']

# Marker and color to use for the plot
marker_per_time_point = { 0 : 'x', 1 : '^', 2 : '*', 3 : '+', 4 : 'D', 5 : 's', 6 : 'o'} # Different marker for each time point
marker_per_time_point = { 0 : 'x', 1 : 'x', 2 : 'x', 3 : '+', 4 : 'o', 5 : 'o', 6 : 'o'} # Different marker for each time point
color_per_group = {'control' : 'blue', 'test_150' : 'green', 'test_300' : 'red'} # Different color for each group
color_per_time_point = { 0 : 'green', 1 : 'green', 2 : 'green', 3 : 'blue', 4 : 'red', 5 : 'red', 6 : 'red'} # Different color for each group

# Prepare the variable for the plot
data_for_wavelength = dict()
timestamp_label = dict()
for group in plant_group_list : 
    data_for_wavelength[group] = []
    timestamp_label[group] = []

# Figure used to plot the data
plt.rcParams.update({'font.size': plot_config['fontsize']})
fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])

# Get the data to plot
for i in range(len(t_list)):
    print(i, t_list[i])

    # Get NIRS data
    path_beans = "data/beans/t{}/csv/beans.csv".format(t_list[i])
    data_beans, _, _, _, _ = manage_data_beans.read_data_beans_single_file(path_beans, return_numpy = False)
    data_beans = data_beans[data_beans['plant'] == plant_to_examine]
    
    # (OPTIONAL) Normalization
    if use_standardization  : data_beans = preprocess.normalize_standardization(data_beans, divide_mems = True)
    if use_sg_preprocess : data_beans = preprocess.sg(data_beans)
    
    for plant_group in plant_group_list:
        print("\t{}".format(plant_group))

        # Get the data for each plant group
        data_group = data_beans[data_beans['test_control'] == plant_group]

        print("\t", len(data_group))

        if len(data_group) > 0 : # Sometimes a group is not recorded or the plants of the group are dead
            for wavelength in wavelength_list:
                tmp_data_wavelegth = data_group.loc[:, str(wavelength)]
                data_for_wavelength.append(tmp_data_wavelegth.to_numpy().T)

            for j in range(len(tmp_data_wavelegth)):
                timestamp_label.append(t_list[i])
                group_label.append(group_label)
            
# Create a matrix of size n. measures x n. wavelengths 
data_to_reduce = np.asarray(data_for_wavelength)

raise ValueError("aaa")
if color_type == 1 :
    ax.scatter(x = tmp_data_wavelegth_1, y = tmp_data_wavelegth_2,
               c = color_per_group[plant_group], marker = marker_per_time_point[t_list[i]], s = 20,
               label = 't{} - {}'.format(t_list[i], plant_group)
               )
elif color_type == 2 :
    ax.scatter(x = tmp_data_wavelegth_1, y = tmp_data_wavelegth_2,
               c = color_per_time_point[t_list[i]], marker = marker_per_time_point[t_list[i]], s = 20,
               label = 't{} - {}'.format(t_list[i], plant_group)
               )
else:
    raise ValueError("color_type can have only 1 or 2 as value")

ax.set_xlabel("wavelength {}nm".format(wavelength_1))
ax.set_ylabel("wavelength {}nm".format(wavelength_2))

ax.grid(True)

if plant_to_examine == 'PhaseolusVulgaris': 
    ax.legend(['control', 'test_150'])


if plant_to_examine == 'ViciaFaba': 
    ax.legend(['control', 'test_150', 'test_300'])

    # if use_standardization : 
    #     ax.set_xlim([-0.98, -0.88])
    #     ax.set_ylim([-1.1, -1.4])


t_string = ""
for t in t_list: t_string += str(t) 

normalization_string = ""
if use_standardization : normalization_string += 'STANDARDIZATION'
if use_sg_preprocess : normalization_string += '_sg'
if not use_standardization and not use_sg_preprocess : normalization_string = 'NO_NORMALIZATION'

ax.set_title("{} - t{} - {}".format(plant_to_examine, t_string, normalization_string))

fig.tight_layout()
fig.show()

if plot_config:
    path_save = 'Saved Results/beans_spectra/scatter_plot_wavelength/'
    os.makedirs(path_save, exist_ok = True)
    

    path_save = 'Saved Results/beans_spectra/scatter_plot_wavelength/'
    path_save += '{}_scatter_wavelength_t{}_{}'.format(plant_to_examine, t_string, normalization_string)
    fig.savefig(path_save + ".png", format = 'png')
