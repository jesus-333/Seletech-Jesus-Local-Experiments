import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

path_dataset = 'data/powder/dataset_whole_PMT.pkl'

mems_to_plot = 1

plot_config = dict(
    figsize = (10, 6),
)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Get the data

powder_dataset = pd.read_pickle(path_dataset)


wavelengts_1 = np.hstack(np.arange(1350, 1650 + 1))
wavelengts_2 = np.arange(1750, 2150 + 1)
wavelength = np.hstack((wavelengts_1, wavelengts_2))

if mems_to_plot == 1 :
    tmp_wave_1 = int(1350)
    tmp_wave_2 = int(1650)
elif mems_to_plot == 2 :
    tmp_wave_1 = int(1750)
    tmp_wave_2 = int(2150)
elif mems_to_plot == 'both' :
    tmp_wave_1 = 1350
    tmp_wave_2 = 2150
else:
    raise ValueError("mems_to_plot must have value 1 or 2 or both")

# Get the wavelenth to plot
wavelength_to_plot = wavelength[np.logical_and(wavelength >= tmp_wave_1, wavelength <= tmp_wave_2)]

# Get data (numpy array)
# powder_data = powder_dataset.loc[:, str(tmp_wave_1):str(tmp_wave_2)].to_numpy()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Plot the data

# fig, ax = plt.subplots(figsize = plot_config['figsize'])
#
# ax.plot(wavelength, powder_dataset.mean(axis = 0), label = 'Mean', color = 'black', linewidth = 2)
# ax.fill_between(wavelength, powder_dataset.mean(axis = 0) - powder_dataset.std(axis = 0), powder_dataset.mean(axis = 0) + powder_dataset.std(axis = 0), alpha = 0.5, color = 'black')
# ax.set_xlabel('Wavelength (nm)')
# ax.set_ylabel('Amplitude')
# ax.grid(True)
#
# fig.tight_layout()
# fig.show()
