# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 22:04:58 2022

@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)
"""

#%% Imports 

import sys
sys.path.insert(0, 'support')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
import time

from support.datasets import load_spectra_data, load_water_data, create_extended_water_vector
from support.timestamp_function import convert_timestamps_in_dataframe
from support.preprocess import aggregate_HT_data_V1, aggregate_HT_data_V2

#%% Load data
normalize_trials = 0

spectra_plants_numpy, wavelength, spectra_timestamp = load_spectra_data("data/[2021-08-05_to_11-26]All_PlantSpectra.csv", normalize_trials)

water_data, water_timestamp = load_water_data("data/[2021-08-05_to_11-26]PlantTest_Notes.csv")
extended_water_timestamp = create_extended_water_vector(water_timestamp, water_data, spectra_timestamp)

ht_data = pd.read_csv("data/[2021-08-05_to_11-26]All_PlantHTSensor.csv", encoding= 'unicode_escape')
humidity = ht_data[' Humidity[%]']
temperature = ht_data[' Temperature[C]']
ht_timestamp = ht_data['Timestamp']

#%% Test spectra division

time_interval_start = 45
time_interval_end = 360

# good_idx, bad_idx = choose_spectra_based_on_water_V1(extended_water_timestamp, time_interval_start = time_interval_start, time_interval_end = time_interval_end)

#%% Aggregate humidity and temperature data to have the same length of spectra array

ht_timestamp = pd.read_csv('data/jesus_ht_timestamp.csv').to_numpy()[:, 1:]
spectra_timestamp_tmp = pd.read_csv('data/jesus_spectra_timestamp.csv').to_numpy()[:, 1:]

h_array = humidity.to_numpy()
t_array = temperature.to_numpy()

a = aggregate_HT_data_V2(ht_timestamp, spectra_timestamp_tmp, h_array, t_array)
aggregate_h_array, aggregate_t_array, aggregate_timestamp = a[0], a[1], a[2]

#%% Plot water timestamp and humidity together

x_axis_array = np.linspace(8 + 5/31, 11 + 26/30, len(aggregate_h_array))

fig, ax = plt.subplots(1, 1, figsize = (15, 10))

ax.plot(x_axis_array, aggregate_h_array, label='Humidity')
ax.set_xlim(x_axis_array[0], x_axis_array[-1])
ax.set_xticks([9, 10, 11])
ax.set_xlabel('Months')
ax.set_ylabel("Humidity [%]")

ax2=ax.twinx()
ax2.plot(x_axis_array, extended_water_timestamp * 50, 'o', color = 'orange', markersize=12, label='Water')
ax2.set_ylim([2, 2.5 * 50])
ax2.set_ylabel("Water Quantity [g]")

h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2, loc=2)

plt.tight_layout()
plt.rcParams.update({'font.size': 22})

#%% Select humidity with standard deviation and plot

n_std = 1

h_mean = np.mean(aggregate_h_array)
h_std = np.std(aggregate_h_array)

idx_to_mantain = aggregate_h_array > h_mean + n_std * h_std
filter_h_array = aggregate_h_array[idx_to_mantain]

x_axis_array = np.linspace(8 + 5/31, 11 + 26/30, len(aggregate_h_array))

fig, ax = plt.subplots(1, 1, figsize = (15, 10))

ax.plot(x_axis_array, aggregate_h_array, label='Humidity')
ax.plot(x_axis_array[idx_to_mantain], aggregate_h_array[idx_to_mantain], 'o',
        label='Humidity above {} std'.format(n_std), markersize = 4, color = 'red')
ax.set_xticks([9, 10, 11])
ax.set_xlabel('Months')
ax.set_ylabel("Humidity [%]")

ax2=ax.twinx()
ax2.plot(x_axis_array, extended_water_timestamp * 50, 'o', color = 'orange', markersize=12, label='Water')
ax2.set_ylim([2, 2.5 * 50])
ax2.set_ylabel("Water Quantity [g]")

h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2, loc=2)

plt.tight_layout()
plt.rcParams.update({'font.size': 22})

#%% Humidity 2 class

def tmp_function_avg_std(spectra):
    spectra_1 = spectra[:, 0:300]
    spectra_2 = spectra[:, -401:-1]
    
    avg_spectra_1 = np.mean(spectra_1, 0)
    std_spectra_1 = np.std(spectra_1)
    
    avg_spectra_2 = np.mean(spectra_2, 0)
    std_spectra_2 = np.std(spectra_2)
    
    return avg_spectra_1, std_spectra_1, avg_spectra_2, std_spectra_2

def tmp_function_plot_axis(ax, x, y, std, color, label):
    ax.plot(x, y, color, label = label)
    ax.plot(x, y + std, color, linestyle='dashed')
    ax.plot(x, y - std, color, linestyle='dashed')
    
    ax.set_xlim([x[0], x[-1]])
    ax.set_ylim([50, 4200])
    
    ax.set_xlabel('Wavelength')
    ax.set_ylabel('Amplitude')
    
    return ax

n_std = 1

wave_1 = wavelength[0:300]
wave_2 = wavelength[-401:-1]

h_mean = np.mean(aggregate_h_array)
h_std = np.std(aggregate_h_array)

idx_to_mantain = aggregate_h_array > h_mean + n_std * h_std
avg_wet_1, std_wet_1, avg_wet_2, std_wet_2 = tmp_function_avg_std(spectra_plants_numpy[idx_to_mantain])
avg_dry_1, std_dry_1, avg_dry_2, std_dry_2 = tmp_function_avg_std(spectra_plants_numpy[np.logical_not(idx_to_mantain)])

fig, ax = plt.subplots(1, 2, figsize = (25, 10))

label_wet = 'Wet (above mean humidity + {} std)'.format(n_std)
label_dry = 'Dry (below mean humidity + {} std)'.format(n_std)

tmp_function_plot_axis(ax[0], wave_1, avg_wet_1, std_wet_1, color = 'blue', label = label_wet)
tmp_function_plot_axis(ax[0], wave_1, avg_dry_1, std_dry_1, color = 'red', label = label_dry)
tmp_function_plot_axis(ax[1], wave_2, avg_wet_2, std_wet_2, color = 'blue', label = label_wet)
tmp_function_plot_axis(ax[1], wave_2, avg_dry_2, std_dry_2, color = 'red', label = label_dry)

ax[0].legend()
ax[1].legend()

plt.tight_layout()
plt.rcParams.update({'font.size': 18})

#%% Humidity 3 class

n_std = 2

wave_1 = wavelength[0:300]
wave_2 = wavelength[-401:-1]

h_mean = np.mean(aggregate_h_array)
h_std = np.std(aggregate_h_array)

idx_1 = aggregate_h_array > h_mean + n_std * h_std
idx_2 = aggregate_h_array < h_mean - n_std * h_std
idx_3 = np.logical_not(np.logical_or(idx_1, idx_2))
avg_wet_1, std_wet_1, avg_wet_2, std_wet_2 = tmp_function_avg_std(spectra_plants_numpy[idx_1])
avg_dry_1, std_dry_1, avg_dry_2, std_dry_2 = tmp_function_avg_std(spectra_plants_numpy[idx_2])
avg_baseline_1, std_baseline_1, avg_baseline_2, std_baseline_2 = tmp_function_avg_std(spectra_plants_numpy[idx_3])

fig, ax = plt.subplots(1, 2, figsize = (25, 10))

label_wet = 'Wet (above mean humidity + {} std)'.format(n_std)
label_dry = 'Dry (below mean humidity - {} std)'.format(n_std)
label_baseline = 'Baseline (Inside  mean humidity ± {} std)'.format(n_std)

tmp_function_plot_axis(ax[0], wave_1, avg_wet_1, std_wet_1, color = 'blue', label = label_wet)
tmp_function_plot_axis(ax[0], wave_1, avg_dry_1, std_dry_1, color = 'red', label = label_dry)
tmp_function_plot_axis(ax[0], wave_1, avg_baseline_1, std_baseline_1, color = 'green', label = label_baseline)
tmp_function_plot_axis(ax[1], wave_2, avg_wet_2, std_wet_2, color = 'blue', label = label_wet)
tmp_function_plot_axis(ax[1], wave_2, avg_dry_2, std_dry_2, color = 'red', label = label_dry)
tmp_function_plot_axis(ax[1], wave_2, avg_baseline_2, std_baseline_2, color = 'green', label = label_baseline)

ax[0].legend()
ax[1].legend()

plt.tight_layout()
plt.rcParams.update({'font.size': 18})
