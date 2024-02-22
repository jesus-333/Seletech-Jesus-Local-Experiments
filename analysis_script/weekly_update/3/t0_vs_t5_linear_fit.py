"""
Script per dei risultati per il 4 meeting.
Confronta il gruppo di controllo a t0 con il gruppo 300ml a t5
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os

from library import manage_data_beans, preprocess

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

lamp_power = 80

# plant = 'PhaseolusVulgaris'
plant = 'ViciaFaba'

# Parameter to remove spectra with amplitude to low
min_amplitude = 1000
percentage_above_threshold = 0.8

# Parameter for preprocess
compute_absorbance = True
use_sg_filter = True
w = 50
p = 3
deriv = 2

mems_to_plot = 1

plot_config = dict(
    figsize = (20, 12),
    fontsize = 20,
    split_plot = False,
    save_fig = True,
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

plt.rcParams.update({'font.size': plot_config['fontsize']})

t_list = [0, 5]
color_list_spectra = ['green', 'red']
color_list_fit = ['darkgreen', 'darkred']

if plot_config['split_plot']:
    fig, axs = plt.subplots(1, 2, figsize = plot_config['figsize'])
else:
    fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])

for i in range(len(t_list)):
    t = t_list[i]

    # Load data
    path_spectra = "data/beans/t{}/csv/beans_avg.csv".format(t)
    spectra_data_ALL, wavelength, group_labels_list, plant_labels_list, plant_type_list = manage_data_beans.read_data_beans_single_file(path_spectra, return_numpy = False)
    spectra_data = spectra_data_ALL[spectra_data_ALL['gain_0'] == 1]
    if plant is not None : spectra_data = spectra_data[spectra_data['plant'] == plant]
    if lamp_power is not None : spectra_data = spectra_data[spectra_data['lamp_0'] == lamp_power]

    # Remove spectra with at least a portion of spectra below the threshold
    spectra_data, idx_data_to_keep = preprocess.filter_spectra_by_threshold(spectra_data, threshold = min_amplitude, percentage_above_threshold = percentage_above_threshold)

    # Preprocess
    meta_data = spectra_data.loc[:, "timestamp":"type"]
    if compute_absorbance : spectra_data = preprocess.R_A(spectra_data, keep_meta = False)
    if use_sg_filter : spectra_data = preprocess.sg(spectra_data, w, p, deriv, keep_meta = False)
    if compute_absorbance or use_sg_filter: # Since during preprocess the metadata are removed here are restored
        spectra_data = pd.concat([meta_data, spectra_data], axis = 1)
    else:
        raise ValueError("At least 1 between compute_absorbance and use_sg_filter must be true")

    if t == 0:
        spectra_data = spectra_data[spectra_data['test_control'] == 'control']
        string_group = 'control (t0)'
    elif t == 5:
        spectra_data = spectra_data[spectra_data['test_control'] == 'test_300']
        string_group = 'test 300 (t5)'
    
    # Select mems to plot
    if mems_to_plot == 1 :
        spectra_data = spectra_data.loc[:, "1400":"1600"].to_numpy()
        tmp_wavelength = wavelength[np.logical_and(wavelength >= 1400, wavelength <= 1600)]
    elif mems_to_plot == 2 :
        spectra_data = spectra_data.loc[:, "1750":"2150"].to_numpy()
        tmp_wavelength = wavelength[wavelength >= 1750]
    else:
        raise ValueError("mems_to_plot must have value 1 or 2")

    specta_mean = spectra_data.mean(0)
    specta_std = spectra_data.std(0)

    # Fit linear regression (y = ax + b)
    linear_reg = LinearRegression().fit(tmp_wavelength.reshape(-1, 1), specta_mean)
    a = linear_reg.coef_
    b = linear_reg.intercept_

    if plot_config['split_plot']: ax = axs[i]
    
    # Plot average and std
    ax.plot(tmp_wavelength, specta_mean, label = string_group, color = color_list_spectra[i])
    ax.fill_between(tmp_wavelength, specta_mean - specta_std, specta_mean + specta_std,
                    color = color_list_spectra[i], alpha = 0.25
                    )

    # Plot linear fit
    ax.plot(tmp_wavelength, tmp_wavelength * a + b, label = 'Linear Fit {}'.format(string_group),
            color = color_list_fit[i], linewidth = 2
            )

    # Other information
    ax.grid(True)
    ax.legend()
    ax.set_xlabel("Wavelength [nm]")
    ax.set_xlim([tmp_wavelength[0], tmp_wavelength[-1]])

    if mems_to_plot == 1:
        ax.set_ylim([-0.6 * 1e-5, 0.3 * 1e-5])
    elif mems_to_plot == 2:
        ax.set_ylim([-1.1 * 1e-5, 1.1 * 1e-5])

fig.tight_layout()
fig.show()

if plot_config['save_fig'] :
    path_save = 'Saved Results/weekly_update_beans/3/t0_vs_t5/'
    os.makedirs(path_save, exist_ok = True)

    if plot_config['split_plot']:
        path_save += '3_t0_vs_t5_linear_fit_w_{}_p_{}_der_{}_mems_{}_split_plot'.format(w, p, deriv, mems_to_plot)
    else:
        path_save += '3_t0_vs_t5_linear_fit_w_{}_p_{}_der_{}_mems_{}_same_plot'.format(w, p, deriv, mems_to_plot)
    fig.savefig(path_save + ".png", format = 'png')
