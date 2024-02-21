"""
@author : Alberto (Jesus) Zancanaro
@organization : University of Padua
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

normalize_hist = True
mems_to_plot = 2

plot_config = dict(
    figsize = (20, 12),
    n_bins = 50,
    fontsize = 20,
    linewidth = 2,
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
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Save data into a DataFrame that can be used by seaborne
    spectra_for_plot = pd.DataFrame(columns = ["plant_group", "wavelength", "amplitude", "lamp"])
    for j in range(len(spectra_data)):
        
        # Get data for the current row
        tmp_row = spectra_data.iloc[j]
        
        # Get data and wavelengths
        if use_sg_filter : # Remove border artifact caused by the filter
            idx_1 = np.logical_and(wavelength >= 1350 + w / 2, wavelength <= 1650 - w / 2)
            idx_2 = np.logical_and(wavelength >= 1750 + w / 2, wavelength <= 2150 - w / 2)

            tmp_wavelength_1 = wavelength[idx_1]
            tmp_wavelength_2 = wavelength[idx_2]
            tmp_wavelength = np.concatenate([tmp_wavelength_1, tmp_wavelength_2], 0)

            tmp_data_1 = tmp_row.loc["1350":"2150"].to_numpy()[idx_1]
            tmp_data_2 = tmp_row.loc["1350":"2150"].to_numpy()[idx_2]
            tmp_data = np.concatenate([tmp_data_1, tmp_data_2], 0)
        else :
            tmp_data = tmp_row.loc["1350":"2150"].to_numpy()
            tmp_wavelength = wavelength.copy()

        # Create list for plant and lamp power
        tmp_plant_group = [tmp_row['test_control']] * len(tmp_data)
        tmp_lamp_power = [tmp_row['lamp_0']] * len(tmp_data)
        
        # Insert data in the DataFrame
        tmp_df = pd.DataFrame(columns = ["plant_group", "wavelength", "amplitude", "lamp"],
                              data = np.asarray([tmp_plant_group, tmp_wavelength, tmp_data, tmp_lamp_power]).T
                              )
        
        spectra_for_plot = pd.concat([spectra_for_plot, tmp_df], axis = 0)

    sns.jointplot(data = spectra_for_plot, x = "wavelength", y = "amplitude", hue = "plant_group")
    plt.xlim([tmp_wavelength[0], tmp_wavelength[-1]])
    plt.grid(True)

    sns.jointplot(data = spectra_for_plot, x = "wavelength", y = "amplitude", hue = "lamp")
    plt.xlim([tmp_wavelength[0], tmp_wavelength[-1]])
    plt.grid(True)

    plt.show()




