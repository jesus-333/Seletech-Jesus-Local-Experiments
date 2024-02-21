"""
Script per dei risultati per il 4 meeting.
Cercare i picchi come ha suggerito Dag.

"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

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
    
    idx_lamp_power = 0
    for j in range(2):
        for k in range(2):
            # Get spectra with specific lamp power
            lamp_power = lamp_power_list[idx_lamp_power]
            idx_lamp_power += 1
            tmp_spectra = spectra_data[spectra_data['lamp_0'] == lamp_power]

            if len(tmp_spectra) > 0:
                # Get average and std per group
                tmp_spectra_mean = tmp_spectra.groupby("test_control").mean(numeric_only = True)
                tmp_spectra_std = tmp_spectra.groupby("test_control").std(numeric_only = True)
                
                # Select mems to plot
                if mems_to_plot == 1 :
                    tmp_spectra_mean = tmp_spectra_mean.loc[:, "1350":"1650"]
                    tmp_wavelength = wavelength[wavelength <= 1650]
                    tmp_wavelength = tmp_wavelength[int(w / 2):-int(w / 2)]
                elif mems_to_plot == 2 :
                    tmp_spectra_mean = tmp_spectra_mean.loc[:, "1750":"2150"]
                    tmp_wavelength = wavelength[wavelength >= 1750]
                    tmp_wavelength = tmp_wavelength[int(w / 2):-int(w / 2)]
                elif mems_to_plot == 'both' :
                    tmp_spectra_mean = tmp_spectra_mean.loc[:, "1350":"2150"]
                    tmp_wavelength = wavelength[:]
                else:
                    raise ValueError("mems_to_plot must have value 1 or 2 or both")

                # Plot the spectra for each group
                for idx_group in range(len(tmp_spectra_mean)):
                    tmp_spectra_mean_numpy = tmp_spectra_mean.iloc[idx_group, :].to_numpy()

                    # Remove border artifacts
                    tmp_spectra_mean_numpy = tmp_spectra_mean_numpy[int(w / 2):-int(w / 2)]
                    
                    # Find the peaks and the corresponding values
                    peaks_idx, _ = find_peaks(tmp_spectra_mean_numpy)
                    peaks_wavelength = tmp_wavelength[peaks_idx]
                    peaks_values = tmp_spectra_mean_numpy[peaks_idx]

