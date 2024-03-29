import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from library import manage_data_beans, preprocess

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
path_calib = 'data/Orange/fruit_orange_calib.csv'
path_NON_calib = 'data/Orange/fruit_orange.csv'

mems_to_use = 1

compute_absorbance = False
use_sg_preprocess = False

w = 50
p = 3
der = 2

use_minmax_norm = True

use_moving_average = False
moving_average_windows = 7

idx = np.random.randint(1920)
idx = 269

plot_config = dict(
    figsize = (12, 8),
    fontsize = 15,
    use_same_figure = False,
    save_fig = False
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

def compute_preprocess(data, compute_absorbance, use_sg_preprocess):
    if compute_absorbance : data  = preprocess.R_A(data, keep_meta = False)
    if use_sg_preprocess : data  = preprocess.sg(data, keep_meta = False, w = w, p = p, deriv = der)

    return data

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

data_calib = pd.read_csv(path_calib)
data_non_calib = pd.read_csv(path_NON_calib)

if compute_absorbance or use_sg_preprocess or use_minmax_norm:
    data_calib = compute_preprocess(data_calib, compute_absorbance, use_sg_preprocess)
    data_non_calib = compute_preprocess(data_non_calib, compute_absorbance, use_sg_preprocess)

# Dato che il codice è stato fatto inizialmente per mems2 nel caso voglia plottare il mems1 scambio le variabili.
# Cioè assegno ai dati del mems1 il nome delle variabili del mems2
if mems_to_use == 1 : 
    data_calib_2, _ = split_data_per_mems(data_calib)
    data_non_calib_2, _ = split_data_per_mems(data_non_calib)

else : 
    _, data_calib_2 = split_data_per_mems(data_calib)
    _, data_non_calib_2 = split_data_per_mems(data_non_calib)

data_calib_2 = data_calib_2[idx]
data_non_calib_2 = data_non_calib_2[idx]

N = moving_average_windows
data_calib_2_smooth = np.convolve(data_calib_2, np.ones(N)/N, mode='valid')
data_non_calib_2_smooth = np.convolve(data_non_calib_2, np.ones(N)/N, mode='valid')

if mems_to_use == 1 : 
    wavelengts_2 = np.arange(1350, 1650 + 1)
    wavelengts_2_smooth = np.linspace(1350, 1650, len(data_calib_2_smooth))

else : 
    wavelengts_2 = np.arange(1750, 2150 + 1)
    wavelengts_2_smooth = np.linspace(1750, 2150, len(data_calib_2_smooth))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Plot spectra and smoothed version together

fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])

ax.plot(wavelengts_2, data_non_calib_2, label = 'Non calib')
# ax.plot(wavelengts_2_smooth, data_non_calib_2_smooth, label = 'Non calib (smooth)', color = 'red')

ax.grid(True)
ax.set_xlabel("Wavelength [nm]")
if compute_absorbance : ax.set_ylabel("Absorbance")
else : ax.set_ylabel("Reflectance")
ax.legend()

fig.tight_layout()
fig.show()

if plot_config['save_fig'] : 
    path_save = 'Saved Results/orange_spectra/'
    os.makedirs(path_save, exist_ok = True)

    if use_sg_preprocess : path_save += 'orange_calibrated_w_{}_p_{}_der_{}_mems2'.format(w, p, der)
    else : path_save += 'orange_calibrated_mems2'
    fig.savefig(path_save + ".png", format = 'png')
    # fig.savefig(path_save + ".pdf", format = 'pdf')

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Plot spectra and smoothed version together

fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])

ax.plot(wavelengts_2, data_calib_2, label = 'Calib')
# ax.plot(wavelengts_2_smooth, data_calib_2_smooth, label = 'Calib (smooth)', color = 'orange')

ax.grid(True)
ax.set_xlabel("Wavelength [nm]")
if compute_absorbance : ax.set_ylabel("Absorbance")
else : ax.set_ylabel("Reflectance")
ax.legend()

fig.tight_layout()
fig.show()

if plot_config['save_fig'] : 
    path_save = 'Saved Results/orange_spectra/'
    os.makedirs(path_save, exist_ok = True)
    
    if use_sg_preprocess : path_save += 'orange_calibrated_w_{}_p_{}_der_{}_mems2'.format(w, p, der)
    else : path_save += 'orange_calibrated_mems2'
    fig.savefig(path_save + ".png", format = 'png')
    # fig.savefig(path_save + ".pdf", format = 'pdf')

    data_calib_2 = ( data_calib_2 - data_calib_2.min() ) / (data_calib_2.max() - data_calib_2.min())
    data_non_calib_2 = ( data_non_calib_2 - data_non_calib_2.min() ) / (data_non_calib_2.max() - data_non_calib_2.min())

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Comparison smoothed version after minmax norm

data_calib_2 = ( data_calib_2 - data_calib_2.min() ) / (data_calib_2.max() - data_calib_2.min())
data_non_calib_2 = ( data_non_calib_2 - data_non_calib_2.min() ) / (data_non_calib_2.max() - data_non_calib_2.min())

data_calib_2_smooth = ( data_calib_2_smooth - data_calib_2_smooth.min() ) / (data_calib_2_smooth.max() - data_calib_2_smooth.min())
data_non_calib_2_smooth = ( data_non_calib_2_smooth - data_non_calib_2_smooth.min() ) / (data_non_calib_2_smooth.max() - data_non_calib_2_smooth.min())

fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])

# ax.plot(wavelengts_2, data_calib_2, label = 'Calib', color = 'red')
# ax.plot(wavelengts_2, data_non_calib_2, label = 'Non calib', color = 'orange')

ax.plot(wavelengts_2_smooth, data_calib_2_smooth, label = 'Calib', color = 'orange')
ax.plot(wavelengts_2_smooth, data_non_calib_2_smooth, label = 'Non calib', color = 'red')

ax.grid(True)
ax.set_xlabel("Wavelength [nm]")
if compute_absorbance : ax.set_ylabel("Absorbance")
else : ax.set_ylabel("Reflectance")
ax.legend()

fig.tight_layout()
fig.show()

if plot_config['save_fig'] : 
    path_save = 'Saved Results/orange_spectra/'
    os.makedirs(path_save, exist_ok = True)
    
    if use_sg_preprocess : path_save += 'orange_calibrated_w_{}_p_{}_der_{}_mems2'.format(w, p, der)
    else : path_save += 'orange_calibrated_mems2'
    fig.savefig(path_save + ".png", format = 'png')
    # fig.savefig(path_save + ".pdf", format = 'pdf')

    data_calib_2 = ( data_calib_2 - data_calib_2.min() ) / (data_calib_2.max() - data_calib_2.min())
    data_non_calib_2 = ( data_non_calib_2 - data_non_calib_2.min() ) / (data_non_calib_2.max() - data_non_calib_2.min())
