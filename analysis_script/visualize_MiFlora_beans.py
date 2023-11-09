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
t_list = [0, 1, 2, 3, 4, 5]

# plant_to_examine = 'ViciaFaba'
# t_list = [0, 1, 2, 3, 4, 6]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
for i in range(len(t_list)):
    print(i)
    path_beans = "data/beans/t{}/csv/beans.csv".format(t_list[i])
    path_MiFlora = "data/beans/t{}/miflora/{}.csv".format(t_list[i], plant_to_examine)

    data_beans, wavelength, group_labels_list, plant_labels_list, plant_type_list = manage_data_beans.read_data_beans_single_file(path_beans, return_numpy = False)

    data_MiFlora = manage_data_other_sensors.read_data_MiFlora(path_MiFlora)
    
    NIRS_timestamps_list = list(data_beans['timestamp'])

    data_MiFlora = manage_data_other_sensors.pair_with_NIRS_sensor_timestamp(NIRS_timestamps_list, data_MiFlora, return_difference = True)

    print(max(data_MiFlora['Difference_with_paired_NIRS_timestamp']))
