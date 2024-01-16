"""

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

use_standardization = False
use_control_group_to_calibrate = False
norm_type_with_control_group = 2 # Used only if use_control_group_to_calibrate == True

compute_absorbance = True
use_sg_preprocess = True

plot_config = dict(
    figsize = (12, 8),
    fontsize = 15,
    save_fig = True
)

use_shaded_area = False

wavelength_to_plot = "1450"
wavelength_to_plot = "1360"

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

if plant_to_examine == 'PhaseolusVulgaris': plant_group_list = ['control', 'test_150',]
else : plant_group_list = ['control', 'test_150', 'test_300']

wavelength_mean_per_type = { 'control' : [], 'test_150' : [], 'test_300' : [] }
wavelength_std_per_type = { 'control' : [], 'test_150' : [], 'test_300' : [] }

lamp_power_list = [50, 60, 70, 80]

mean_to_plot = { 'control' : [], 'test_150' : [], 'test_300' : [] }
std_to_plot  = { 'control' : [], 'test_150' : [], 'test_300' : [] }

color_per_group = {'control' : 'blue', 'test_150' : 'green', 'test_300' : 'red'}
marker_per_group = {'control' : 'x', 'test_150' : 'o', 'test_300' : 's'}
linestyle_per_group = {'control' : 'solid', 'test_150' : 'dashdot', 'test_300' : 'dashed'}

for i in range(len(t_list)):
    # Get NIRS data
    path_beans = "data/beans/t{}/csv/beans.csv".format(t_list[i])
    data_beans, wavelength, group_labels_list, plant_labels_list, plant_type_list = manage_data_beans.read_data_beans_single_file(path_beans, return_numpy = False)
    data_beans = data_beans[data_beans['plant'] == plant_to_examine]
    
    # Remove measure with 0 gain (it is a single measure per plant)
    data_beans = data_beans[data_beans['gain_0'] != 0]


    for plant_group in plant_group_list :
        # Get specific group
        data_beans_group = data_beans[data_beans['test_control'] == plant_group]

        wavelength_mean_per_lamp_power = []
        wavelength_std_per_lamp_power  = []
        for lamp_power in lamp_power_list :
            # Get specific lamp poweer
            data_lamp_power = data_beans_group[data_beans_group['lamp_0'] == lamp_power]

            if len(data_lamp_power) > 0 : 
                # Normalize data
                if use_standardization  : data_lamp_power  = preprocess.normalize_standardization(data_lamp_power, divide_mems = True)
                if use_control_group_to_calibrate : data_lamp_power  = preprocess.normalize_with_control_group(data_lamp_power, norm_type = norm_type_with_control_group)

                if compute_absorbance : data_lamp_power = preprocess.R_A(data_lamp_power) 
                if use_sg_preprocess : data_lamp_power  = preprocess.sg(data_lamp_power)

                # Get wavelength
                wavelength_mean = data_lamp_power.loc[:, wavelength_to_plot].mean()
                wavelength_std = data_lamp_power.loc[:, wavelength_to_plot].std()

                wavelength_mean_per_lamp_power.append(wavelength_mean)
                wavelength_std_per_lamp_power.append(wavelength_std)

        if len(wavelength_mean_per_lamp_power) == 4:
            wavelength_mean_per_type[plant_group].append(wavelength_mean_per_lamp_power)
            wavelength_std_per_type[plant_group].append(wavelength_mean_per_lamp_power)
        else:
            # wavelength_mean_per_type[plant_group].append(wavelength_mean_per_type[plant_group][-1])
            # wavelength_std_per_type[plant_group].append(wavelength_std_per_type[plant_group][-1])

            wavelength_mean_per_type[plant_group].append(np.ones(4) * np.nan)
            wavelength_std_per_type[plant_group].append(np.ones(4) * np.nan)

#%% Normalize time series
for plant_group in plant_group_list : 
    # Convert in numpy array
    wavelength_mean_per_type[plant_group] = np.asarray(wavelength_mean_per_type[plant_group]).T

    # Compute mean (along time series)(i.e. bring the curves for different lamp power around 0)
    mean_per_lamp_power = np.nanmean(wavelength_mean_per_type[plant_group], 1)

    # Remove mean by every time series
    wavelength_mean_per_type[plant_group] = (wavelength_mean_per_type[plant_group].T - mean_per_lamp_power ).T

    # Compute mean and std (along lamp power)
    mean_to_plot[plant_group] = np.nanmean(wavelength_mean_per_type[plant_group], 0)
    std_to_plot[plant_group] = np.nanstd(wavelength_mean_per_type[plant_group], 0)

#%% Plot the data
plt.rcParams.update({'font.size': plot_config['fontsize']})
fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])
for plant_group in plant_group_list :
    # Convert in numpy array
    mean_to_plot[plant_group] = np.asarray( mean_to_plot[plant_group] )
    std_to_plot[plant_group] = np.asarray( std_to_plot[plant_group] )

    if use_shaded_area:
        ax.plot(t_list, mean_to_plot[plant_group],
                    label = plant_group, marker = 'o', color = color_per_group[plant_group]
                    )
        ax.fill_between(t_list, mean_to_plot[plant_group] + std_to_plot[plant_group],
                        mean_to_plot[plant_group] - std_to_plot[plant_group],
                        alpha = 0.25, color = color_per_group[plant_group]
                        )
    else:
        ax.errorbar(t_list, mean_to_plot[plant_group], std_to_plot[plant_group],
                    label = plant_group,  capsize = 8,color = color_per_group[plant_group],
                    marker = marker_per_group[plant_group], markersize = 10,
                    linestyle = linestyle_per_group[plant_group]
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
