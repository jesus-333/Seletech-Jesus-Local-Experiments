import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from library import manage_data_beans, preprocess

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

path_data_NON_calib = 'data/Orange/fruit_orange.csv'
path_data_calib = 'data/Orange/fruit_orange_calib.csv'

add_moving_average = False
moving_average_windows = 5

w = 50
p = 3
deriv = 2

compute_absorbance = True
use_sg_preprocess = True

compute_absorbance = False
use_sg_preprocess = False

idx = np.random.randint(1920)
idx = 666

plot_config = dict(
    figsize = (12, 8),
    save_fig = True,
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def compute_preprocess(data, compute_absorbance, use_sg_preprocess):
    if compute_absorbance : 
        tmp_data = preprocess.R_A(data.copy(), keep_meta = False) 
        data.loc[:, "1350":"2150"] = tmp_data
    if use_sg_preprocess : 
        tmp_data  = preprocess.sg(data.copy(), w = w , p = p, deriv = deriv, keep_meta = False)
        data.loc[:, "1350":"2150"] = tmp_data

    return data

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

data_calib = pd.read_csv(path_data_calib)
data_NON_calib = pd.read_csv(path_data_NON_calib)

if compute_absorbance or use_sg_preprocess:
    data_calib = compute_preprocess(data_calib, compute_absorbance, use_sg_preprocess)
    data_NON_calib = compute_preprocess(data_NON_calib, compute_absorbance, use_sg_preprocess)


spectra_calib = data_calib.iloc[idx]
spectra_NON_calib = data_NON_calib.iloc[idx]

spectra_calib_mems_1 = spectra_calib.loc["1350":"1650"].to_numpy()
spectra_calib_mems_2 = spectra_calib.loc["1750":"2150"].to_numpy()

spectra_NON_calib_mems_1 = spectra_NON_calib.loc["1350":"1650"].to_numpy()
spectra_NON_calib_mems_2 = spectra_NON_calib.loc["1750":"2150"].to_numpy()

wavelength_1 = np.arange(1350, 1650 + 1)
wavelength_2 = np.arange(1750, 2150 + 1)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# (OPTIONAL) Moving average

if add_moving_average :
    N = moving_average_windows
    moving_average_NON_calib_mems_1 = np.convolve(spectra_NON_calib_mems_1, np.ones(N)/N, mode='valid')
    moving_average_NON_calib_mems_2 = np.convolve(spectra_NON_calib_mems_2, np.ones(N)/N, mode='valid')

    moving_average_calib_mems_1 = np.convolve(spectra_calib_mems_1, np.ones(N)/N, mode='valid')
    moving_average_calib_mems_2 = np.convolve(spectra_calib_mems_2, np.ones(N)/N, mode='valid')

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

fig, axs = plt.subplots(2, 2, figsize = plot_config['figsize'])

axs[0, 0].plot(wavelength_1, spectra_NON_calib_mems_1, label = 'NON calib')
axs[1, 0].plot(wavelength_1, spectra_calib_mems_1, label = 'calib')

axs[0, 1].plot(wavelength_2, spectra_NON_calib_mems_2, label = 'NON calib')
axs[1, 1].plot(wavelength_2, spectra_calib_mems_2, label = 'calib')

if add_moving_average:
    wavelength_1 = np.linspace(1350, 1650, len(moving_average_calib_mems_1))
    wavelength_2 = np.linspace(1750, 2150, len(moving_average_calib_mems_2))

    # axs[0, 0].plot(wavelength_1, moving_average_NON_calib_mems_1, label = 'NON calib', alpha = 0.8)
    # axs[1, 0].plot(wavelength_1, moving_average_calib_mems_1, label = 'calib', alpha = 0.8)

    axs[0, 1].plot(wavelength_2, moving_average_NON_calib_mems_2, label = 'NON calib', alpha = 0.8)
    axs[1, 1].plot(wavelength_2, moving_average_calib_mems_2, label = 'calib', alpha = 0.8)

axs[0, 0].set_title("MEMS 1 - Non calibrated")
axs[1, 0].set_title("MEMS 1 - Calibrated")

axs[0, 1].set_title("MEMS 2 - Non calibrated")
axs[1, 1].set_title("MEMS 2 - Calibrated")

for i in range(2):
    for j in range(2):
        ax = axs[i, j]
        ax.grid(True)
        ax.set_xlabel("wavelength [nm]")



title = "fruit_orange data" 
if use_sg_preprocess :
    title += " - w = {}, p = {}, der = {}".format(w, p, deriv)

fig.suptitle(title)
fig.tight_layout()
fig.show()

if plot_config['save_fig']:
    path_save = 'Saved Results/beans_spectra/single_spectra_orange/'
    os.makedirs(path_save, exist_ok = True)

    path_save += 'calibraton_orange_idx_{}_w_{}_p_{}_der_{}'.format(idx, w, p, deriv)
    fig.savefig(path_save + ".png", format = 'png')

