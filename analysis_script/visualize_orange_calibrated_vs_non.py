import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

path_data_NON_calib = 'data/Orange/fruit_orange.csv'
path_data_calib = 'data/Orange/fruit_orange_calib.csv'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

data_calib = pd.read_csv(path_data_calib)
data_NON_calib = pd.read_csv(path_data_NON_calib)

idx = np.random.randint(len(data_calib))

spectra_calib = data_calib.iloc[idx]
spectra_NON_calib = data_NON_calib.iloc[idx]

spectra_calib_mems_1 = spectra_calib.loc["1350":"1650"].to_numpy()
spectra_calib_mems_2 = spectra_calib.loc["1750":"2150"].to_numpy()

spectra_NON_calib_mems_1 = spectra_NON_calib.loc["1350":"1650"].to_numpy()
spectra_NON_calib_mems_2 = spectra_NON_calib.loc["1750":"2150"].to_numpy()

wavelength_1 = np.arange(1350, 1650 + 1)
wavelength_2 = np.arange(1750, 2150 + 1)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

fig, axs = plt.subplots(2, 2, figsize = (12, 8))

axs[0, 0].plot(wavelength_1, spectra_NON_calib_mems_1, label = 'NON calib')
axs[1, 0].plot(wavelength_1, spectra_calib_mems_1, label = 'calib')

axs[0, 1].plot(wavelength_2, spectra_NON_calib_mems_2, label = 'NON calib')
axs[1, 1].plot(wavelength_2, spectra_calib_mems_2, label = 'calib')


axs[0, 0].set_title("MEMS 1 - Non calibrated")
axs[1, 0].set_title("MEMS 2 - Non calibrated")

axs[0, 1].set_title("MEMS 1 - Calibrated")
axs[1, 1].set_title("MEMS 2 - Calibrated")

for i in range(2):
    for j in range(2):
        ax = axs[i, j]
        ax.grid(True)
        ax.set_xlabel("wavelength [nm]")

fig.tight_layout()
fig.show()
