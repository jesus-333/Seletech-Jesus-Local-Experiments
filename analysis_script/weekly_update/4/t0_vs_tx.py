"""
Script per dei risultati per il 4 meeting.
Confronta il gruppo di controllo a t0 con il gruppo 300ml a tx dove x è un giorno tra 0 e 5

"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

from library import manage_data_beans, preprocess

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

plant = 'PhaseolusVulgaris'
plant = 'ViciaFaba'

lamp_power = 80

t_comparison = 5

# Parameter to remove spectra with amplitude to low
min_amplitude = 1000
percentage_above_threshold = 0.8

# Parameter for preprocess
compute_absorbance = True
use_SNV = True
use_sg_filter = True
w = 20
p = 3
deriv = 2

mems_to_plot = 1

plot_config = dict(
    figsize = (20, 12),
    fontsize = 20,
    xlim = [1400, 1600],
    # xlim = None
    split_plot = False,
    save_fig = True,
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

group_color = {'test_300' : 'red', 'control' : 'green', 'test_150' : 'blue'}
group_linestyle = {'test_300' : 'dashed', 'control' : 'solid', 'test_150' : 'dashdot'}
group_linewidth = {'test_300' : 4, 'control' : 1, 'test_150' : 2}

plt.rcParams.update({'font.size': plot_config['fontsize']})

t_list = [0, t_comparison]
linestyle_list = ['solid', 'dashed']
color_list = ['green', 'red']

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
    if use_SNV : spectra_data = preprocess.normalize_standardization(spectra_data, divide_mems = True)
    if use_sg_filter : spectra_data = preprocess.sg(spectra_data, w, p, deriv, keep_meta = False)
    if compute_absorbance or use_sg_filter: # Since during preprocess the metadata are removed here are restored
        spectra_data = pd.concat([meta_data, spectra_data], axis = 1)
    else:
        raise ValueError("At least 1 between compute_absorbance and use_sg_filter must be true")
    
    # Select control group for t0 or test_300 group for t5
    if t == 0:
        group = 'control'
        string_group = 'control (t0)'
    elif t == t_comparison:
        if plant == 'ViciaFaba' :
            group = 'test_300'
            string_group = 'test 300 (t{})'.format(t_comparison)
        else :
            group = 'test_150'
            string_group = 'test 150 (t{})'.format(t_comparison)
    spectra_data = spectra_data[spectra_data['test_control'] == group]

    # Select mems to plot
    if mems_to_plot == 1 :
        # Originale
        # spectra_data = spectra_data.loc[:, "1350":"1650"].to_numpy()
        # tmp_wavelength = wavelength[np.logical_and(wavelength >= 1350, wavelength <= 1650)]

        tmp_wave_1 = int(1350 + w / 2)
        tmp_wave_2 = int(1650 - w / 2)
    elif mems_to_plot == 2 :
        # Originale
        # spectra_data = spectra_data.loc[:, "1750":"2150"].to_numpy()
        # tmp_wavelength = wavelength[wavelength >= 1750]

        tmp_wave_1 = int(1750 + w / 2)
        tmp_wave_2 = int(2150 - w / 2)
    elif mems_to_plot == 'both' :
        tmp_wave_1 = 1350
        tmp_wave_2 = 2150
    else:
        raise ValueError("mems_to_plot must have value 1 or 2 or both")


    spectra_data = spectra_data.loc[:, str(tmp_wave_1):str(tmp_wave_2)].to_numpy()
    tmp_wavelength = wavelength[np.logical_and(wavelength >= tmp_wave_1, wavelength <= tmp_wave_2)]

    specta_mean = spectra_data.mean(0)
    specta_std = spectra_data.std(0)

    if plot_config['split_plot']: ax = axs[i]

    ax.plot(tmp_wavelength, specta_mean, label = string_group,
            color = group_color[group], linestyle = group_linestyle[group], linewidth = group_linewidth[group],
            )
    ax.fill_between(tmp_wavelength, specta_mean - specta_std, specta_mean + specta_std,
                    color = group_color[group], linestyle = group_linestyle[group], linewidth = group_linewidth[group],
                    alpha = 0.25
                    )

    ax.grid(True)
    ax.legend()
    ax.set_xlabel("Wavelength [nm]")
    if plot_config['xlim'] is not None : ax.set_xlim([plot_config['xlim'][0], plot_config['xlim'][1]])
    else : ax.set_xlim([tmp_wavelength[0], tmp_wavelength[-1]])

    if mems_to_plot == 1:
        if use_SNV :
            if plant == 'ViciaFaba' :
                if w == 15 :
                    ax.set_ylim([-1 * 1e-3, 0.8 * 1e-3])
                if w == 50 :
                    ax.set_ylim([-0.32 * 1e-3, 0.12 * 1e-3])
            else :
                ax.set_ylim([-0.45 * 1e-3, 0.3 * 1e-3])
        else:
            ax.set_ylim([-0.6 * 1e-5, 0.3 * 1e-5])
    elif mems_to_plot == 2:
        if use_SNV :
            if plant == 'ViciaFaba' :
                pass
            else :
                ax.set_ylim([-1.5 * 1e-3, 1.5 * 1e-3])
        else:
            ax.set_ylim([-1.1 * 1e-5, 1.1 * 1e-5])

fig.tight_layout()
fig.show()

if plot_config['save_fig'] :
    path_save = 'Saved Results/weekly_update_beans/4/t0_vs_tx/'
    os.makedirs(path_save, exist_ok = True)

    if plot_config['split_plot']:
        path_save += '3_t0_vs_t{}_w_{}_p_{}_der_{}_mems_{}_split_plot'.format(t_comparison, w, p, deriv, mems_to_plot)
    else:
        path_save += '3_t0_vs_t{}_w_{}_p_{}_der_{}_mems_{}_same_plot'.format(t_comparison, w, p, deriv, mems_to_plot)
    fig.savefig(path_save + ".png", format = 'png')
