"""
Compute the std of a specific wavelength in 2 scenario (given a specific lamp power, gain and group).
1) Same plant in different days
2) Different plants the same day

It is used to check if the variation between days is greater than the variation between plants 
"""

import numpy as np
import matplotlib.pyplot as plt

from library import preprocess, manage_data_beans

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

plant_to_examine = 'PhaseolusVulgaris'
plant_to_examine = 'ViciaFaba'

lamp_power_spectra_to_plot = 60
gain_spectra_to_plot = 1
group_spectra_to_plot = 'test_300' # Possible value are control, test_150, test_300

use_standardization = False
use_control_group_to_calibrate = True
norm_type_with_control_group = 2 # Used only if use_control_group_to_calibrate == True
use_sg_preprocess = False

normalize_per_lamp_power = True # If true normalize each group of lamp power separately

group_spectra_to_analyze = 'test_300'
wavelength_to_analyze = 1450

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# List of plants inside the group
if group_spectra_to_analyze == 'test_150' : 
    n_plant_list = ['NACL150_1', 'NACL150_2', 'NACL150_3']
    t_list = [0, 1, 2, 3, 4, 5, 6]
if group_spectra_to_analyze  == 'test_300' : 
    n_plant_list = ['NACL300_1', 'NACL300_2', 'NACL300_3']
    t_list = [0, 1, 2, 3, 4, 5]

# Matrix to save the wavelength
wavelength_matrix = np.zeros((len(t_list), 3))

# Get the data for the computation 
for i in range(len(t_list)): # Cycle between days
    # Get NIRS data
    path_beans = "data/beans/t{}/csv/beans.csv".format(t_list[i])
    data_beans_full, wavelength, group_labels_list, plant_labels_list, plant_type_list = manage_data_beans.read_data_beans_single_file(path_beans, return_numpy = False)
    data_beans = data_beans_full[data_beans_full['plant'] == plant_to_examine]

    # Normalization
    if normalize_per_lamp_power :
        lamp_power_list = [50, 60, 70, 80]
        for lamp_power in lamp_power_list :
            data_lamp_power = data_beans[data_beans['lamp_0'] == lamp_power]

            if use_standardization : data_lamp_power = preprocess.normalize_standardization(data_lamp_power, divide_mems = True)
            if use_control_group_to_calibrate : data_lamp_power = preprocess.normalize_with_control_group(data_lamp_power, norm_type = norm_type_with_control_group)
            if use_sg_preprocess : data_lamp_power = preprocess.sg(data_lamp_power)

            data_beans[data_beans['lamp_0'] == lamp_power] = data_lamp_power
    else :
        if use_standardization : data_beans = preprocess.normalize_standardization(data_beans, divide_mems = True)
        if use_control_group_to_calibrate : data_beans = preprocess.normalize_with_control_group(data_beans, norm_type = norm_type_with_control_group)
        if use_sg_preprocess : data_beans = preprocess.sg(data_beans)
    
    # Filter data for the group
    data_beans = data_beans[data_beans['test_control'] == group_spectra_to_plot]
    if len(data_beans) > 0:

        # Filter for gain and lamp power
        data_beans = data_beans[data_beans['gain_0'] == gain_spectra_to_plot]
        data_beans = data_beans[data_beans['lamp_0'] == lamp_power_spectra_to_plot]
        for j in range(len(n_plant_list)) : # Cycle between plants
            n_plant = n_plant_list[j]
            data_to_analyze = data_beans[data_beans['type'] == n_plant].loc[:, str(wavelength_to_analyze)].mean()
            
            wavelength_matrix[i, j] = data_to_analyze
            

