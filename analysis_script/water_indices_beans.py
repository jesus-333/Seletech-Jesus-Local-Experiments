"""
Compute and plot water indices for the beans experiment

@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

import numpy as np
import matplotlib.pyplot as plt

from library import manage_data_other_sensors, manage_data_beans, timestamp_functions
from library import preprocess

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Settings 

plant_to_examine = 'PhaseolusVulgaris'
plant_to_examine = 'ViciaFaba'
t_list = [0, 1, 2, 3, 4, 5, 6]

plot_config = dict(
    figsize = (12, 8),
    fontsize = 15,
    save_fig = True
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

plt.rcParams.update({'font.size': plot_config['fontsize']})
fig, axs = plt.subplots(1, 1, figsize = plot_config['figsize'])

plant_group_list = ['control', 'test_150', 'test_300']
water_idx_per_type = dict(
    control = [],
    test_150 = [],
    test_300 = []
)

for i in range(len(t_list)):
    # Get NIRS data
    path_beans = "data/beans/t{}/csv/beans.csv".format(t_list[i])
    data_beans, wavelength, group_labels_list, plant_labels_list, plant_type_list = manage_data_beans.read_data_beans_single_file(path_beans, return_numpy = False)
    data_beans = data_beans[data_beans['plant'] == plant_to_examine]
    
    # Preprocess and average by plant tpe
    data_beans_reflactance = preprocess.sg(data_beans)
    data_beans_absorbance = preprocess.R_A(data_beans_reflactance)
    grouped_data = data_beans_absorbance.groupby('test_control')

    # Compute NDNI
    A_1510 = grouped_data['1510'].mean()
    A_1680 = grouped_data['1650'].mean()
    ndni_list = (A_1510 - A_1680) / (A_1510 + A_1680)

    for plant_group in plant_group_list :
        if plant_group in ndni_list:
            water_idx_per_type[plant_group].append(ndni_list[plant_group])
        else:
            water_idx_per_type[plant_group].append(water_idx_per_type[plant_group][-1])

plt.rcParams.update({'font.size': plot_config['fontsize']})
fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])
for plant_group in plant_group_list :
    ax.plot(t_list, water_idx_per_type[plant_group], label = plant_group, marker = 'o')

ax.set_xlabel("Time point")
ax.set_ylabel("NDNI")
ax.legend()
ax.grid()
fig.show()
