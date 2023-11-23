"""
Visualize the data collected through the MiFlora during the beans experiment

@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

import numpy as np
import matplotlib.pyplot as plt

from library import manage_data_other_sensors, manage_data_beans, timestamp_functions

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
#%% Get the data

if plant_to_examine == 'PhaseolusVulgaris': 
    plant_group_list = ['control', 'test_150',]
    timestamp_conversion_mode  = [1, 1, 1, 1, 1, 2, 1]
else : 
    plant_group_list = ['control', 'test_150', 'test_300']
    timestamp_conversion_mode = [2, 1, 1, 1, 1, 2, 1]


color_per_group = {'control' : 'blue', 'test_150' : 'green', 'test_300' : 'red'}
marker_per_group = {'control' : 'x', 'test_150' : 'o', 'test_300' : 's'}
linestyle_per_group = {'control' : 'solid', 'test_150' : 'dashdot', 'test_300' : 'dashed'}

conductivity_per_plant_group = dict()
for group in plant_group_list: conductivity_per_plant_group[group] = []

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

    for plant_group in plant_group_list:
        data_beans = data_beans_full[data_beans_full['plant'] == plant_to_examine]
        data_beans = data_beans[data_beans['test_control'] == plant_group]
        
        # Pair MiFlora data with NIRS data
        NIRS_timestamps_list = list(data_beans['timestamp'])
        data_MiFlora = manage_data_other_sensors.pair_with_NIRS_sensor_timestamp(NIRS_timestamps_list, data_MiFlora_full, return_difference = True)
        
        # Compute and save average conductivity
        # average_conductivity = data_MiFlora['MI_CONDUCTIVITY'].to_numpy(dtype = float).mean()
        average_conductivity = data_MiFlora['MI_MOISTURE'].to_numpy(dtype = float).mean()
        conductivity_per_plant_group[plant_group].append(average_conductivity)
        # print(average_conductivity)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Plot the data

plt.rcParams.update({'font.size': plot_config['fontsize']})
fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])
for plant_group in plant_group_list:
    ax.plot(t_list, conductivity_per_plant_group[plant_group], 
            label = plant_group, linestyle = linestyle_per_group[plant_group],
            marker = marker_per_group[plant_group], markersize = 10,
            )

ax.grid(True)
ax.legend()
ax.set_xlabel("Time points")
ax.set_ylabel("Conductivity")
ax.set_title("{} - Conductivity MiFlora".format(plant_to_examine))

fig.tight_layout()
fig.show()
