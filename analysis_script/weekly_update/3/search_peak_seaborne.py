"""
Script per dei risultati per il 4 meeting.
Cercare i picchi come ha suggerito Dag.

Cerca i picchi su tutti i dati di un giorno senza distinzione di gruppo/lamp power e calcola alcune statistiche di base.

"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from library import manage_data_beans, preprocess

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

t_list = [0, 1, 2, 3, 4, 5, 6]
# t_list = [0]

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
peaks_min_distance = 30

mems_to_plot = 1

plot_config = dict(
    figsize = (20, 12),
    fontsize = 20,
    add_std = True,
    save_fig = True,
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

plt.rcParams.update({'font.size': plot_config['fontsize']})
sns.set(rc={'figure.figsize': plot_config['figsize']})

lamp_power_list = [50, 60, 70, 80]
peaks_df = pd.DataFrame(columns = ["plant_group", "wavelength", "amplitude", "lamp", "day", "point_type"])

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

    for idx_spectra in range(len(spectra_data)):
        tmp_row = spectra_data.iloc[idx_spectra, :]
        tmp_spectra_numpy = tmp_row.to_numpy()

        # Select mems to plot
        if mems_to_plot == 1 :
            tmp_spectra_numpy = tmp_row.loc["1350":"1650"].to_numpy()
            tmp_wavelength = wavelength[wavelength <= 1650]
            tmp_wavelength = tmp_wavelength[int(w / 2):-int(w / 2)]
        elif mems_to_plot == 2 :
            tmp_spectra_numpy = tmp_row.loc["1750":"2150"].to_numpy()
            tmp_wavelength = wavelength[wavelength >= 1750]
            tmp_wavelength = tmp_wavelength[int(w / 2):-int(w / 2)]
        elif mems_to_plot == 'both' :
            tmp_spectra_numpy = tmp_row.loc["1350":"2150"].to_numpy()
            tmp_wavelength = wavelength[:]
        else:
            raise ValueError("mems_to_plot must have value 1 or 2 or both")

        # Remove border artifacts
        tmp_spectra_numpy = tmp_spectra_numpy[int(w / 2):-int(w / 2)]
        
        # Find the peaks and the corresponding values
        peaks_idx, _ = find_peaks(tmp_spectra_numpy, width = peaks_width_in_samples, distance = peaks_min_distance)
        peaks_wavelength = tmp_wavelength[peaks_idx]
        peaks_values = tmp_spectra_numpy[peaks_idx]

        valley_idx, _ = find_peaks(-tmp_spectra_numpy, width = peaks_width_in_samples, distance = peaks_min_distance)
        valley_wavelength = tmp_wavelength[peaks_idx]
        valley_values = tmp_spectra_numpy[peaks_idx]

        valley_wavelength = []
        valley_values = []

        # peaks_values_list += list(peaks_values)
        # peaks_wavelength_list += list(peaks_wavelength)
        
        # Merge peak and valley
        tmp_point_list = np.concatenate((peaks_values, valley_values), axis = 0)
        tmp_wavelength_list = np.concatenate((peaks_wavelength, valley_wavelength), axis = 0)
        tmp_point_type_list = ['peak'] * len(peaks_values) + ['valley'] * len(valley_values)

        # Create list for plant and lamp power
        tmp_plant_group = [tmp_row['test_control']] * len(tmp_point_list)
        tmp_lamp_power = [tmp_row['lamp_0']] * len(tmp_point_list)
        tmp_day = np.ones(len(tmp_point_list)) * t

        tmp_df = pd.DataFrame(columns = ["plant_group", "wavelength", "amplitude", "lamp", "day", "point_type"],
                              data = np.asarray([tmp_plant_group, tmp_wavelength_list, tmp_point_list, tmp_lamp_power, tmp_day, tmp_point_type_list]).T
                              )

        peaks_df = pd.concat([peaks_df, tmp_df], axis = 0)


sns_plot = sns.jointplot(data = peaks_df, x = "wavelength", y = "amplitude", hue = "plant_group",
              xlim = [tmp_wavelength[0], tmp_wavelength[-1]], height = 15,
              )
sns_plot.savefig("peaks_seaborne_plant.png")

sns_plot = sns.jointplot(data = peaks_df, x = "wavelength", y = "amplitude", hue = "lamp",
              xlim = [tmp_wavelength[0], tmp_wavelength[-1]], height = 15, 
              )
sns_plot.savefig("peaks_seaborne_lamp.png")

sns_plot = sns.jointplot(data = peaks_df, x = "wavelength", y = "amplitude", hue = "day",
              xlim = [tmp_wavelength[0], tmp_wavelength[-1]], height = 15,
              )
sns_plot.savefig("peaks_seaborne_day.png")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Plot example of peaks detection

fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])

ax.plot(tmp_wavelength, tmp_spectra_numpy, color = 'black')
ax.plot(peaks_wavelength, peaks_values, 'x', color = 'red', markersize = 15)
ax.plot(tmp_wavelength[valley_idx], tmp_spectra_numpy[valley_idx], 'x', color = 'green', markersize = 15)

ax.set_xlim([tmp_wavelength[0], tmp_wavelength[-1]])
ax.set_xlabel("Wavelength [nm]")
ax.set_ylabel("Amplitude")

plt.grid(True)
plt.tight_layout()
plt.show()

