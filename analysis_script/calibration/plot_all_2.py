import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from library import manage_data_beans, preprocess

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

t = 0

# plant = 'PhaseolusVulgaris'
plant = 'ViciaFaba'

remove_mean = False

plot_config = dict(
    figsize = (20, 10),
    fontsize = 15,
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

plt.rcParams.update({'font.size': plot_config['fontsize']})

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
# Load data 

# Calibration data
path_xtalk_file = "data/xtalk_wcs_SRS.csv"
calibration_data = pd.read_csv(path_xtalk_file)
calibration_data['target'] = calibration_data['target'].str.lower()

# Spectra data
path_spectra = "data/beans/t{}/csv/beans_avg.csv".format(t)
spectra_data, wavelength, group_labels_list, plant_labels_list, plant_type_list = manage_data_beans.read_data_beans_single_file(path_spectra, return_numpy = False)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

wavelengts_1 = np.arange(1350, 1650 + 1)
wavelengts_2 = np.arange(1750, 2150 + 1)

lamp_power_list = list(set(calibration_data['lamp_0']))
lamp_power_list.sort()

gain_list = [0, 1]

fig_1, axs_1 = plt.subplots(2, 3, figsize = plot_config['figsize'])
fig_2, axs_2 = plt.subplots(2, 3, figsize = plot_config['figsize'])

for i in range(len(gain_list)):
    gain = gain_list[i]
    for j in range(len(lamp_power_list)):
        lamp_power = lamp_power_list[j]
        
        # Filter by lamp power and gain (Calibration data)
        tmp_calibration_data = calibration_data[calibration_data['gain_0'] == gain]
        tmp_calibration_data = tmp_calibration_data[tmp_calibration_data['lamp_0'] == lamp_power]

        # Filter by lamp power and gain (Spectra data)
        tmp_spectra_data = spectra_data[spectra_data['gain_0'] == gain]
        tmp_spectra_data = tmp_spectra_data[tmp_spectra_data['lamp_0'] == lamp_power]

        # Get xtalk, srs, wcs and spectra data in separate variables for each mems 
        srs_data_1, srs_data_2 = split_data_per_mems(tmp_calibration_data[tmp_calibration_data['target'] == 'srs'], remove_mean)
        wcs_data_1, wcs_data_2 = split_data_per_mems(tmp_calibration_data[tmp_calibration_data['target'] == 'wcs'], remove_mean)
        crosstalk_data_1, crosstalk_data_2 = split_data_per_mems(tmp_calibration_data[tmp_calibration_data['target'] == 'crosstalk'], remove_mean)
        spectra_data_1, spectra_data_2 = split_data_per_mems(tmp_spectra_data, remove_mean)

        # Plot mems 1
        axs_1[i, 0].plot(wavelengts_1, srs_data_1, label = "lamp {}".format(lamp_power))
        axs_1[i, 1].plot(wavelengts_1, crosstalk_data_1, label = "lamp {}".format(lamp_power))
        axs_1[i, 2].plot(wavelengts_1, spectra_data_1.T, label = "lamp {}".format(lamp_power))
        axs_1[i, 0].set_title("SRS - gain {}".format(gain))
        axs_1[i, 1].set_title("Xtalk - gain {}".format(gain))
        axs_1[i, 2].set_title("Spectra - gain {}".format(gain))

        # Plot mems 2
        axs_2[i, 0].plot(wavelengts_2, srs_data_2, label = "lamp {}".format(lamp_power))
        axs_2[i, 1].plot(wavelengts_2, crosstalk_data_2, label = "lamp {}".format(lamp_power))
        axs_2[i, 2].plot(wavelengts_2, spectra_data_2.T, label = "lamp {}".format(lamp_power))
        axs_2[i, 0].set_title("SRS - gain {}".format(gain))
        axs_2[i, 1].set_title("Xtalk - gain {}".format(gain))
        axs_2[i, 2].set_title("Spectra - gain {}".format(gain))

for i in range(2):
    for j in range(3):
        axs_1[i, j].set_xlabel('Wavelength [nm]')
        axs_1[i, j].grid(True)
        # axs_1[i, j].legend()

        axs_2[i, j].set_xlabel('Wavelength [nm]')
        axs_2[i, j].grid(True)
        # axs_2[i, j].legend(fontsize = 7)

fig_1.tight_layout()
fig_1.show()

fig_2.tight_layout()
fig_2.show()
