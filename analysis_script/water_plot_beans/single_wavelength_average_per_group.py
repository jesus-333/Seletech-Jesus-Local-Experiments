"""
For the beans experiment plot across time the 1450nm and the 1950nm wavelength
For each group (control, test_150, test_300) we take the AVERAGE of the measures.
Each group has different plants inside

@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

import numpy as np
import matplotlib.pyplot as plt
import os

from library import manage_data_other_sensors, manage_data_beans, timestamp_functions
from library import preprocess

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Settings 

plant_to_examine = 'PhaseolusVulgaris'
plant_to_examine = 'ViciaFaba'
t_list = [0, 1, 2, 3, 4, 5, 6]
# t_list = [0]

normalize_first_value = False
use_standardization = True
use_control_group_to_calibrate = False
norm_type_with_control_group = 2 # Used only if use_control_group_to_calibrate == True
use_sg_preprocess = False

normalize_per_lamp_power = True # If true normalize each group of lamp power separately

normalize_time_series = True # If True normalize the series of point t0, t1 etc

plot_config = dict(
    figsize = (12, 8),
    fontsize = 15,
    save_fig = True
)

use_shaded_area = False

wavelength_to_plot = "1450"
# wavelength_to_plot = "2075"

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

plt.rcParams.update({'font.size': plot_config['fontsize']})
fig, axs = plt.subplots(1, 1, figsize = plot_config['figsize'])

if plant_to_examine == 'PhaseolusVulgaris': plant_group_list = ['control', 'test_150',]
else : plant_group_list = ['control', 'test_150', 'test_300']
wavelength_mean_per_type = dict(
    control = [],
    test_150 = [],
    test_300 = []
)

wavelength_std_per_type = dict(
    control = [],
    test_150 = [],
    test_300 = []
)

color_per_group = {'control' : 'blue', 'test_150' : 'green', 'test_300' : 'red'}

for i in range(len(t_list)):
    # Get NIRS data
    path_beans = "data/beans/t{}/csv/beans.csv".format(t_list[i])
    data_beans, wavelength, group_labels_list, plant_labels_list, plant_type_list = manage_data_beans.read_data_beans_single_file(path_beans, return_numpy = False)
    data_beans = data_beans[data_beans['plant'] == plant_to_examine]
    
    # Remove measure with 0 gain (it is a single measure per plant)
    # data_beans = data_beans[data_beans['gain_0'] != 0]

    # Preprocess and average by plant tpe
    if normalize_per_lamp_power :
        lamp_power_list = [50, 60, 70, 80]
        for lamp_power in lamp_power_list :
            data_lamp_power = data_beans[data_beans['lamp_0'] == lamp_power]

            if normalize_first_value : data_lamp_power = preprocess.normalize_with_values_first_column(data_lamp_power, divide_mems = True)
            if use_standardization  : data_lamp_power = preprocess.normalize_standardization(data_lamp_power, divide_mems = True)
            if use_control_group_to_calibrate : data_lamp_power = preprocess.normalize_with_control_group(data_lamp_power, norm_type = norm_type_with_control_group)
            if use_sg_preprocess : data_lamp_power = preprocess.sg(data_lamp_power)

            data_beans[data_beans['lamp_0'] == lamp_power] = data_lamp_power

    else :
        if normalize_first_value : data_beans = preprocess.normalize_with_values_first_column(data_beans, divide_mems = True)
        if use_standardization  : data_beans = preprocess.normalize_standardization(data_beans, divide_mems = True)
        if use_control_group_to_calibrate : data_beans = preprocess.normalize_with_control_group(data_beans, norm_type = norm_type_with_control_group)
        if use_sg_preprocess : data_beans = preprocess.sg(data_beans)

    grouped_data = data_beans.groupby('test_control')

    # Get wavelength
    wavelength_mean = grouped_data[wavelength_to_plot].mean()
    wavelength_std = grouped_data[wavelength_to_plot].std()

    for plant_group in plant_group_list :
        if plant_group in wavelength_mean:
            wavelength_mean_per_type[plant_group].append(wavelength_mean[plant_group])
            wavelength_std_per_type[plant_group].append(wavelength_std[plant_group])
        else:
            wavelength_mean_per_type[plant_group].append(wavelength_mean_per_type[plant_group][-1])
            wavelength_std_per_type[plant_group].append(wavelength_std_per_type[plant_group][-1])

plt.rcParams.update({'font.size': plot_config['fontsize']})
fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])
for plant_group in plant_group_list :
    # Convert in numpy array
    wavelength_mean_per_type[plant_group] = np.asarray( wavelength_mean_per_type[plant_group] )
    wavelength_std_per_type[plant_group] = np.asarray( wavelength_std_per_type[plant_group] )

    # Normalize time series (remove mean and divide by std)
    if normalize_time_series :
        time_series_mean = wavelength_mean_per_type[plant_group].mean()
        time_series_std = wavelength_std_per_type[plant_group].std()

        wavelength_mean_per_type[plant_group] = (wavelength_mean_per_type[plant_group] - time_series_mean) / time_series_std
        # wavelength_std_per_type[plant_group] = (wavelength_std_per_type[plant_group] - time_series_mean) / time_series_std
        wavelength_std_per_type[plant_group] = (wavelength_std_per_type[plant_group]) / time_series_std
        std_to_use = time_series_std
    else:
        std_to_use = wavelength_std_per_type[plant_group],


    if use_shaded_area:
        ax.plot(t_list, wavelength_mean_per_type[plant_group],
                    label = plant_group, marker = 'o', color = color_per_group[plant_group]
                    )
        ax.fill_between(t_list, wavelength_mean_per_type[plant_group] + wavelength_std_per_type[plant_group],
                        wavelength_mean_per_type[plant_group] - wavelength_std_per_type[plant_group],
                        alpha = 0.25, color = color_per_group[plant_group]
                        )
    else:
        ax.errorbar(t_list, wavelength_mean_per_type[plant_group], std_to_use,
                    label = plant_group, marker = 'o', capsize = 8,color = color_per_group[plant_group]
                    )

ax.set_xlabel("Time point")
ax.set_ylabel("Amplitude")
ax.set_title("Wavelength {} evolution - {}".format(wavelength_to_plot, plant_to_examine))
ax.legend()
ax.grid()

fig.tight_layout()
fig.show()


if plot_config['save_fig']:
    path_save = 'Saved Results/beans_spectra/'
    os.makedirs(path_save, exist_ok = True)

    path_save = 'Saved Results/beans_spectra/'
    path_save += '{}_wavelength_{}_per_group_beans'.format(plant_to_examine, wavelength_to_plot)
    fig.savefig(path_save + ".png", format = 'png')
    # fig.savefig(path_save + ".pdf", format = 'pdf')

# TODO remove
data_numpy = data_beans.loc[:, "1350":"1650"].to_numpy()
