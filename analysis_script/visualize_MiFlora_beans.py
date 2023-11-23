"""
Visualize the data collected through the MiFlora during the beans experiment

@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

import numpy as np
import matplotlib.pyplot as plt
import os

from library import manage_data_other_sensors, manage_data_beans, timestamp_functions

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Settings 

plant_to_examine = 'PhaseolusVulgaris'
plant_to_examine = 'ViciaFaba'

t_list = [0, 1, 2, 3, 4, 5, 6]
# t_list = [5]

type_data_to_plot = 'MI_MOISTURE'
type_data_to_plot = 'MI_CONDUCTIVITY'

plot_config = dict(
    figsize = (12, 8),
    fontsize = 15,
    save_fig = False
)

mi_flora_to_use = None
mi_flora_to_use = 'Green'
mi_flora_to_use = 'White'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Get the data

if plant_to_examine == 'PhaseolusVulgaris': 
    plant_group_list = ['control', 'test_150',]
    timestamp_conversion_mode  = [1, 1, 1, 1, 1, 2, 1]
else : 
    plant_group_list = ['control', 'test_150', 'test_300']
    plant_group_list = ['test_150']
    timestamp_conversion_mode = [2, 1, 1, 1, 1, 2, 1]


color_per_group = {'control' : 'blue', 'test_150' : 'green', 'test_300' : 'red'}
marker_per_group = {'control' : 'x', 'test_150' : 'o', 'test_300' : 's'}
linestyle_per_group = {'control' : 'solid', 'test_150' : 'dashdot', 'test_300' : 'dashed'}

mi_flora_data_per_plant_group = dict()
for group in plant_group_list: mi_flora_data_per_plant_group[group] = []

for i in range(len(t_list)):
    print(i)

    path_beans = "data/beans/t{}/csv/beans.csv".format(t_list[i])
    path_MiFlora = "data/beans/t{}/miflora/{}.csv".format(t_list[i], plant_to_examine)
    
    # Get NIRS data
    data_beans_full, wavelength, group_labels_list, plant_labels_list, plant_type_list = manage_data_beans.read_data_beans_single_file(path_beans, return_numpy = False)

    # Remove measure with 0 gain (it is a single measure per plant)
    data_beans_full = data_beans_full[data_beans_full['gain_0'] != 0]
    
    # Get MiFlora data 
    data_MiFlora_full = manage_data_other_sensors.read_data_MiFlora(path_MiFlora, timestamp_conversion_mode = timestamp_conversion_mode[i])
    if mi_flora_to_use is not None : 
        id_device_list = list(set(data_MiFlora_full['DEVICEID']))
        idx_id_device = 0 if mi_flora_to_use in id_device_list[0] else 1
        data_MiFlora_full = data_MiFlora_full[data_MiFlora_full['DEVICEID'] == id_device_list[idx_id_device]]

    for plant_group in plant_group_list:
        data_beans = data_beans_full[data_beans_full['plant'] == plant_to_examine]
        data_beans = data_beans[data_beans['test_control'] == plant_group]
        
        # Pair MiFlora data with NIRS data
        NIRS_timestamps_list = list(data_beans['timestamp'])
        data_MiFlora = manage_data_other_sensors.pair_with_NIRS_sensor_timestamp(NIRS_timestamps_list, data_MiFlora_full, return_difference = True)
        
        # Compute and save average conductivity
        average_mi_flora_data = data_MiFlora[type_data_to_plot].to_numpy(dtype = float).mean()
        mi_flora_data_per_plant_group[plant_group].append(average_mi_flora_data)
        # print(average_conductivity)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Plot the data

plt.rcParams.update({'font.size': plot_config['fontsize']})
fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])
for plant_group in plant_group_list:
    ax.plot(t_list, mi_flora_data_per_plant_group[plant_group], 
            label = plant_group, linestyle = linestyle_per_group[plant_group],
            marker = marker_per_group[plant_group], markersize = 10,
            )

ax.grid(True)
ax.legend()
ax.set_xlabel("Time points")
ax.set_ylabel(type_data_to_plot)
ax.set_title("{} - {} MiFlora".format(plant_to_examine, type_data_to_plot))

fig.tight_layout()
fig.show()

if plot_config['save_fig']:
    path_save = 'Saved Results/beans_spectra/mi_flora/'
    os.makedirs(path_save, exist_ok = True)

    path_save += 'data_{}'.format(type_data_to_plot)
    fig.savefig(path_save + ".png", format = 'png')
    # fig.savefig(path_save + ".pdf", format = 'pdf')
