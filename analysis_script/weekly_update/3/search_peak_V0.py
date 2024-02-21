"""
Script per dei risultati per il 4 meeting.
Cercare i picchi come ha suggerito Dag.

Cerca i picchi su tutti i dati di un giorno senza distinzione di gruppo/lamp power e calcola alcune statistiche di base.

"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from library import manage_data_beans, preprocess

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

t_list = [0, 1, 2, 3, 4, 5, 6]
t_list = [0]

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

peaks_width_in_samples = None
peaks_min_distance = 6

mems_to_plot = 2

plot_config = dict(
    figsize = (20, 12),
    fontsize = 20,
    add_std = True,
    save_fig = True,
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

plt.rcParams.update({'font.size': plot_config['fontsize']})

lamp_power_list = [50, 60, 70, 80]

for i in range(len(t_list)):
    # Variable to save the data
    peaks_wavelength_list = []
    peaks_values_list = []
    
    # Get day
    t = t_list[i]

    # Load data
    path_spectra = "data/beans/t{}/csv/beans_avg.csv".format(t)
    spectra_data_ALL, wavelength, group_labels_list, plant_labels_list, plant_type_list = manage_data_beans.read_data_beans_single_file(path_spectra, return_numpy = False)
    spectra_data_ALL = spectra_data_ALL[spectra_data_ALL['gain_0'] == 1]
    if plant is not None : spectra_data = spectra_data_ALL[spectra_data_ALL['plant'] == plant]

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

    # Select mems to plot
    if mems_to_plot == 1 :
        tmp_spectra = spectra_data.loc[:, "1350":"1650"]
        tmp_wavelength = wavelength[wavelength <= 1650]
        tmp_wavelength = tmp_wavelength[int(w / 2):-int(w / 2)]
    elif mems_to_plot == 2 :
        tmp_spectra = spectra_data.loc[:, "1750":"2150"]
        tmp_wavelength = wavelength[wavelength >= 1750]
        tmp_wavelength = tmp_wavelength[int(w / 2):-int(w / 2)]
    elif mems_to_plot == 'both' :
        tmp_spectra = spectra_data.loc[:, "1350":"2150"]
        tmp_wavelength = wavelength[:]
    else:
        raise ValueError("mems_to_plot must have value 1 or 2 or both")

    # Plot the spectra for each group
    for idx_spectra in range(len(tmp_spectra)):
        tmp_spectra_numpy = tmp_spectra.iloc[idx_spectra, :].to_numpy()

        # Remove border artifacts
        tmp_spectra_numpy = tmp_spectra_numpy[int(w / 2):-int(w / 2)]
        
        # Find the peaks and the corresponding values
        peaks_idx, _ = find_peaks(tmp_spectra_numpy, width = peaks_width_in_samples, distance = peaks_min_distance)
        # peaks_idx = np.diff(np.sign(np.diff(tmp_spectra_numpy))).nonzero()[0] + 1
        peaks_wavelength = tmp_wavelength[peaks_idx]
        peaks_values = tmp_spectra_numpy[peaks_idx]

        peaks_values_list += list(peaks_values)
        peaks_wavelength_list += list(peaks_wavelength)

    fig_wavelength, ax_wavelength = plt.subplots(1, 1, figsize = plot_config['figsize'])
    ax_wavelength.hist(peaks_wavelength_list, bins = len(peaks_wavelength_list))
    ax_wavelength.grid(True)
    ax_wavelength.set_xlabel("wavelength [nm]")
    ax_wavelength.set_xlim([tmp_wavelength[0], tmp_wavelength[-1]])
    fig_wavelength.tight_layout()
    fig_wavelength.show()

    fig_values, ax_values = plt.subplots(1, 1, figsize = plot_config['figsize'])
    ax_values.hist(peaks_values_list, bins = len(peaks_values_list))
    ax_values.grid(True)
    ax_values.set_xlabel("Amplitude")
    fig_values.tight_layout()
    fig_values.show()
