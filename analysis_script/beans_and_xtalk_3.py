# Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from library import manage_data_beans, preprocess

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Settings
t = 0

lamp_power_to_plot = 50
gain = 1

# plant = 'PhaseolusVulgaris'
plant = 'ViciaFaba'

percentage_reflectance_srs = 0.99

compute_absorbance = False
use_sg_preprocess = False

w = 50
p = 3
der = 2

use_minmax_norm = True

idx_spectra = 3

plot_config = dict(
    figsize = (12, 8),
    fontsize = 15,
    use_same_figure = False,
    save_fig = False
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def split_data_per_mems(data_dataframe, remove_mean = False):
    data_mems_1 = data_dataframe.loc[:, "1350":"1650"].to_numpy().squeeze()
    data_mems_2 = data_dataframe.loc[:, "1750":"2150"].to_numpy().squeeze()

    if remove_mean : 
        if len(data_mems_1.shape) > 1:
            data_mems_1 = ( data_mems_1.T - data_mems_1.mean(1) ).T
            data_mems_2 = ( data_mems_2.T - data_mems_2.mean(1) ).T
        else : 
            data_mems_1 = data_mems_1 - data_mems_1.mean()
            data_mems_2 = data_mems_2 - data_mems_2.mean()

    return data_mems_1, data_mems_2

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Create wavelength arrays
wavelengts_1 = np.hstack(np.arange(1350, 1650 + 1))
wavelengts_2 = np.arange(1750, 2150 + 1)

# Get data (calibration)
path_xtalk_file = "data/xtalk_wcs_SRS.csv"
n_measure_per_type_xtalk = 11
calibration_data = pd.read_csv(path_xtalk_file)

# Get data (spectra)
path_spectra = "data/beans/t{}/csv/beans_avg.csv".format(t)
spectra_data, wavelength, group_labels_list, plant_labels_list, plant_type_list = manage_data_beans.read_data_beans_single_file(path_spectra, return_numpy = False)

# Filter data (both) according to settings
spectra_data = spectra_data[spectra_data['plant'] == plant]
spectra_data = spectra_data[spectra_data['gain_0'] == gain]
spectra_data = spectra_data[spectra_data['lamp_0'] == lamp_power_to_plot]

calibration_data = calibration_data[calibration_data['gain_0'] == gain]
calibration_data = calibration_data[calibration_data['lamp_0'] == lamp_power_to_plot]

# Check if we have no data for the settings
if len(spectra_data) == 0 : raise ValueError("There are no spectra data with these settings")
if len(calibration_data) == 0 : raise ValueError("There are no calibration data with these settings")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Compare spectra with calibration data

# Get xtalk, srs and wcs in separate array
calibration_data['target'] = calibration_data['target'].str.lower()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Normalized spectra with calibration data

spectra_data_calib = preprocess.normalize_with_srs_and_xtalk(spectra_data, calibration_data[calibration_data['target'] == 'srs'], calibration_data[calibration_data['target'] == 'crosstalk'], percentage_reflectance_srs)


# Create figure
fig, ax = plt.subplots(1, 1, figsize = (12, 8))

# NON Calibrated data
spectra_data_1, _  = split_data_per_mems(spectra_data, False)
ax.plot(wavelengts_1, spectra_data_1[idx_spectra], label = 'Non calibrated', color = 'red')
ax.legend()
ax.set_xlabel("Wavelength [nm]")
ax.grid(True)
fig.tight_layout()
fig.show()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

fig, ax = plt.subplots(1, 1, figsize = (12, 8))

# Calibrated data
spectra_data_calib_1, _  = split_data_per_mems(spectra_data_calib, False)
ax.plot(wavelengts_1, spectra_data_calib_1[idx_spectra], label = 'Calibrated', color = 'orange')
ax.legend()
ax.set_xlabel("Wavelength [nm]")
ax.grid(True)
fig.tight_layout()
fig.show()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

spectra_data_1 = ( spectra_data_1 - spectra_data_1.min() ) / (spectra_data_1.max() - spectra_data_1.min())
spectra_data_calib_1 = ( spectra_data_calib_1 - spectra_data_calib_1.min() ) / (spectra_data_calib_1.max() - spectra_data_calib_1.min())

fig, ax = plt.subplots(1, 1, figsize = (12, 8))

# Calibrated data
ax.plot(wavelengts_1, spectra_data_1[idx_spectra], label = 'Non Calibrated', color = 'red')
ax.plot(wavelengts_1, spectra_data_calib_1[idx_spectra], label = 'Calibrated', color = 'orange')
ax.legend()
ax.set_xlabel("Wavelength [nm]")
ax.grid(True)
fig.tight_layout()
fig.show()
