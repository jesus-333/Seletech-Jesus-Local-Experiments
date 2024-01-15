import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

path_xtalk_file = "data/xtalk_wcs_SRS.csv"
n_measure_per_type = 11

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

calibration_data = pd.read_csv(path_xtalk_file)

wavelengts_1 = np.hstack(np.arange(1350, 1650 + 1))
wavelengts_2 = np.arange(1750, 2150 + 1)

for i in range(6):
    data_to_plot = calibration_data.iloc[i*n_measure_per_type:(i + 1) * n_measure_per_type, :] 
    data_type = data_to_plot.iloc[0, 0]
    lamp_power_list = data_to_plot["lamp_0"].to_numpy() 
    data_to_plot_1 = data_to_plot.loc[:, "1350":"1650"].to_numpy()
    data_to_plot_2 = data_to_plot.loc[:, "1750":"2150"].to_numpy()

    print(i, set(data_to_plot['target']))

    if i == 3:
        row_to_skip = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif i == 4:
        row_to_skip = []

    fig, axs = plt.subplots(1, 2, figsize = (12, 10))
    
    axs[0].plot(wavelengts_1, data_to_plot_1.T, label = lamp_power_list)
    axs[1].plot(wavelengts_2, data_to_plot_2.T, label = lamp_power_list)

    for ax in axs: 
        ax.legend()
        ax.set_title(data_type + " " + str(i))
        ax.set_xlabel("Wavelength [nm]")

    fig.tight_layout()
    fig.show()

