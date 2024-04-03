"""
Create and plot a time series of wavelength, i.e. take a specific wavelength for each day, create the time series and plot it
Optionally you could take a range around the wavelength and plot the time series of the average

@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import os

from library import manage_data_beans
from library import preprocess

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Settings

plant_to_examine = 'PhaseolusVulgaris'
plant_to_examine = 'ViciaFaba'
lamp_power = 80

# Parameter to remove spectra with amplitude to low
min_amplitude = 1000
percentage_above_threshold = 0.8

# Parameter for preprocess
compute_absorbance = True
use_SNV = True
use_sg_filter = True
w = 30
p = 3
deriv = 2

normalize_time_series = False # If True normalize the time series of the wavelength (i.e. a wavelength for each day)

plot_config = dict(
    figsize = (12, 8),
    fontsize = 15,
    use_real_distance = True,
    # ylim = [1600, 2300],
    save_fig = True,
)

use_shaded_area = False

wavelength_to_plot = 1530
# wavelength_to_plot = 1575

average_range = 10 # N. of wavelength to use on left and right to compute the average

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

t_list = [0, 1, 2, 3, 4, 5, 6]
t_day_elapsed = [0, 1, 2, 3, 8, 15, 22] # Remember that the data are taken after different interval of time


# t_list = [0, 1, 2, 3, 4, 5, ]
# t_day_elapsed = [0, 1, 2, 3, 8, 15,] # Remember that the data are taken after different interval of time

if plant_to_examine == 'PhaseolusVulgaris': plant_group_list = ['control', 'test_150',]
else : 
    plant_group_list = ['control', 'test_150', 'test_300']
    # plant_group_list = ['control', 'test_300']

wavelength_mean_per_group = { 'control' : [], 'test_150' : [], 'test_300' : [] }
wavelength_std_per_group = { 'control' : [], 'test_150' : [], 'test_300' : [] }

mean_to_plot = { 'control' : [], 'test_150' : [], 'test_300' : [] }
std_to_plot  = { 'control' : [], 'test_150' : [], 'test_300' : [] }

color_per_group = {'control' : 'blue', 'test_150' : 'green', 'test_300' : 'red'}
marker_per_group = {'control' : 'x', 'test_150' : 'o', 'test_300' : 's'}
linestyle_per_group = {'control' : 'solid', 'test_150' : 'dashdot', 'test_300' : 'dashed'}

p_value_array = np.zeros((len(t_list), len(plant_group_list), len(plant_group_list)))

for i in range(len(t_list)):
    t = t_list[i]

    # Load data
    path_spectra = "data/beans/t{}/csv/beans_avg.csv".format(t)
    spectra_data_ALL, wavelength, group_labels_list, plant_labels_list, plant_type_list = manage_data_beans.read_data_beans_single_file(path_spectra, return_numpy = False)
    spectra_data = spectra_data_ALL[spectra_data_ALL['gain_0'] == 1]
    spectra_data = spectra_data[spectra_data['plant'] == plant_to_examine]
    if lamp_power is not None : spectra_data = spectra_data[spectra_data['lamp_0'] == lamp_power]

    # Remove spectra with at least a portion of spectra below the threshold
    spectra_data, idx_data_to_keep = preprocess.filter_spectra_by_threshold(spectra_data, threshold = min_amplitude, percentage_above_threshold = percentage_above_threshold)

    # Preprocess
    meta_data = spectra_data.loc[:, "timestamp":"type"]
    if compute_absorbance : spectra_data = preprocess.R_A(spectra_data, keep_meta = False)
    if use_SNV : spectra_data = preprocess.normalize_standardization(spectra_data, divide_mems = True)
    if use_sg_filter : spectra_data = preprocess.sg(spectra_data, w, p, deriv, keep_meta = False)
    if compute_absorbance or use_sg_filter: # Since during preprocess the metadata are removed here are restored
        spectra_data = pd.concat([meta_data, spectra_data], axis = 1)

    for j in range(len(plant_group_list)) :
        plant_group = plant_group_list[j]

        # Get specific group
        data_beans_group = spectra_data[spectra_data['test_control'] == plant_group]

        # Get wavelength
        wavelength_mean_1 = data_beans_group.loc[:, str(wavelength_to_plot - average_range):str(wavelength_to_plot + average_range)].mean(0)
        wavelength_std_1 = data_beans_group.loc[:, str(wavelength_to_plot - average_range):str(wavelength_to_plot + average_range)].std()

        # Compute mean and std
        wavelength_mean = wavelength_mean_1.mean()
        wavelength_std = wavelength_std_1.mean()

        # wavelength_mean_per_group.append(wavelength_mean)
        # wavelength_std_per_group.append(wavelength_std)

        wavelength_mean_per_group[plant_group].append(wavelength_mean)
        wavelength_std_per_group[plant_group].append(wavelength_std)

        # Compute t-test for the specific day
        for k in range(len(plant_group_list)) :
            # Get data for the other groups
            data_beans_group_2 = spectra_data[spectra_data['test_control'] == plant_group_list[k]]
            wavelength_mean_2 = data_beans_group_2.loc[:, str(wavelength_to_plot - average_range):str(wavelength_to_plot + average_range)].mean(0)
            wavelength_std_2 = data_beans_group_2.loc[:, str(wavelength_to_plot - average_range):str(wavelength_to_plot + average_range)].std()

            # Compute t-test
            # print(plant_group_list[j],wavelength_mean_1)
            # print(plant_group_list[k],wavelength_mean_2, "\n")
            t_test_output = ttest_ind(wavelength_mean_1.to_numpy(), wavelength_mean_2.to_numpy(), equal_var = False)
            t_statistics, p_value = t_test_output.statistic, t_test_output.pvalue

            p_value_array[i, j, k] = p_value

# Removed unused variables
# del wavelength_mean_1, wavelength_std_1
# del data_beans_group_2, wavelength_mean_2, wavelength_std_2

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Normalize time series

if normalize_time_series:
    print("Time series IS normalized")
    for plant_group in plant_group_list : # TODO check
        pass
        # # Convert in numpy array
        # wavelength_mean_per_group[plant_group] = np.asarray(wavelength_mean_per_group[plant_group]).T
        #
        # # Compute mean (along time series)(i.e. bring the curves for different lamp power around 0)
        # mean_per_group = np.nanmean(wavelength_mean_per_group[plant_group], 1)
        #
        # # Remove mean by every time series
        # wavelength_mean_per_group[plant_group] = (wavelength_mean_per_group[plant_group].T - mean_per_lamp_power ).T
        #
        # # Compute mean and std (along lamp power)
        # mean_to_plot[plant_group] = np.nanmean(wavelength_mean_per_group[plant_group], 0)
        # std_to_plot[plant_group] = np.nanstd(wavelength_mean_per_group[plant_group], 0)
else :
    print("Time series IS NOT normalized")

    for plant_group in plant_group_list :
        mean_to_plot[plant_group] = wavelength_mean_per_group[plant_group]
        std_to_plot[plant_group] = wavelength_std_per_group[plant_group]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Plot the data

plt.rcParams.update({'font.size': plot_config['fontsize']})
fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])

if plot_config['use_real_distance'] : t_plot = t_day_elapsed
else : t_plot = t_list

for plant_group in plant_group_list :
    # Convert in numpy array
    mean_to_plot[plant_group] = np.asarray(mean_to_plot[plant_group])
    std_to_plot[plant_group] = np.asarray(std_to_plot[plant_group])

    if use_shaded_area:
        ax.plot(t_plot, mean_to_plot[plant_group],
                label = plant_group, marker = 'o', color = color_per_group[plant_group]
                )
        ax.fill_between(t_plot, mean_to_plot[plant_group] + std_to_plot[plant_group],
                        mean_to_plot[plant_group] - std_to_plot[plant_group],
                        alpha = 0.25, color = color_per_group[plant_group]
                        )
    else:
        ax.errorbar(t_plot, mean_to_plot[plant_group], std_to_plot[plant_group],
                    label = plant_group, capsize = 8, color = color_per_group[plant_group],
                    marker = marker_per_group[plant_group], markersize = 10,
                    linestyle = linestyle_per_group[plant_group]
                    )

ax.set_xlabel("Time point")
ax.set_ylabel("Amplitude")
# ax.set_title("Wavelength {} ± {} evolution - {}".format(wavelength_to_plot, average_range, plant_to_examine))
ax.set_title("Band of interest {}nm - {}nm ({})".format(wavelength_to_plot - average_range, wavelength_to_plot + average_range, plant_to_examine))
if 'ylim' in plot_config : ax.set_ylim(plot_config['ylim'])
ax.legend()
ax.grid()

fig.tight_layout()
fig.show()


if plot_config['save_fig']:
    path_save = 'Saved Results/weekly_update_beans/4/time_series_wavelength/'
    os.makedirs(path_save, exist_ok = True)

    path_save += '{}_wavelength_{}_average_{}'.format(plant_to_examine, wavelength_to_plot, average_range)
    if compute_absorbance : path_save += '_RA'
    if use_SNV : path_save += '_SNV'
    if use_sg_filter : path_save += '_w_{}_p_{}_der_{}'.format(w, p, deriv)
    fig.savefig(path_save + ".png", format = 'png')
    # fig.savefig(path_save + ".pdf", format = 'pdf')


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# t-test entire time series

for plant_group_1 in plant_group_list :
    # Convert in numpy array
    mean_group_1 = np.asarray(mean_to_plot[plant_group_1])[0:6]

    for plant_group_2 in plant_group_list :
        # Convert in numpy array
        mean_group_2 = np.asarray(mean_to_plot[plant_group_2])[0:6]

        # Compute t-test
        t_test_output = ttest_ind(mean_group_1, mean_group_2, equal_var = False)
        t_statistics, p_value = t_test_output.statistic, t_test_output.pvalue

        print("p-value for time series {} vs {} : {}".format(plant_group_1, plant_group_2, p_value))
