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

remove_mean = True

percentage_reflectance_srs = 0.99

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
srs_data_1, srs_data_2 = split_data_per_mems(calibration_data[calibration_data['target'] == 'srs'], remove_mean)
wcs_data_1, wcs_data_2 = split_data_per_mems(calibration_data[calibration_data['target'] == 'wcs'], remove_mean)
crosstalk_data_1, crosstalk_data_2 = split_data_per_mems(calibration_data[calibration_data['target'] == 'crosstalk'], remove_mean)

# Get sample data for each catagory of spectra data
data_control_1, data_control_2 = split_data_per_mems(spectra_data[spectra_data['test_control'] == 'control'], remove_mean)
data_test_150_1, data_test_150_2 = split_data_per_mems(spectra_data[spectra_data['test_control'] == 'test_150'], remove_mean)
data_test_300_1, data_test_300_2 = split_data_per_mems(spectra_data[spectra_data['test_control'] == 'test_300'], remove_mean)
idx_spectra = np.random.randint(5)

# Create figure
fig, axs = plt.subplots(1, 2, figsize = (12, 8))

# Spectra (mems 1)
axs[0].plot(wavelengts_1, data_control_1[idx_spectra], label = 'control')
axs[0].plot(wavelengts_1, data_test_150_1[idx_spectra], label = 'test_150')
axs[0].plot(wavelengts_1, data_test_300_1[idx_spectra], label = 'test_300')

# Calibartion data (mems 1)
axs[0].plot(wavelengts_1, srs_data_1, label = 'srs')
axs[0].plot(wavelengts_1, crosstalk_data_1, label = 'crosstalk')

# Spectra (mems 2)
axs[1].plot(wavelengts_2, data_control_2[idx_spectra], label = 'control')
axs[1].plot(wavelengts_2, data_test_150_2[idx_spectra], label = 'test_150')
axs[1].plot(wavelengts_2, data_test_300_2[idx_spectra], label = 'test_300')

# Calibartion data (mems 2)
axs[1].plot(wavelengts_2, srs_data_2, label = 'srs')
axs[1].plot(wavelengts_2, crosstalk_data_2, label = 'crosstalk')

for ax in axs:
    ax.legend()
    ax.set_xlabel("Wavelength [nm]")
    ax.grid(True)

fig.suptitle("Spectra at t{}".format(t))
fig.tight_layout()
fig.show()