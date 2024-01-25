import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

path_xtalk_file = "data/xtalk_wcs_SRS.csv"
n_measure_per_type = 11

lamp_power_to_plot = [80]
lamp_power_to_plot = None 

gain = 0

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

if gain == 0 : idx_gain = [0, 1, 2]
elif gain == 1 : idx_gain = [3, 4, 5]
else : raise ValueError("gain must be 0 or 1")

calibration_data = pd.read_csv(path_xtalk_file)

wavelengts_1 = np.hstack(np.arange(1350, 1650 + 1))
wavelengts_2 = np.arange(1750, 2150 + 1)

for i in range(len(idx_gain)):
    tmp_idx_gain = idx_gain[i]
    data_to_plot = calibration_data.iloc[tmp_idx_gain*n_measure_per_type:(tmp_idx_gain + 1) * n_measure_per_type, :] 
    data_type = data_to_plot.iloc[0, 0]
    lamp_power_list = data_to_plot["lamp_0"].to_numpy() 
    data_to_plot_1 = data_to_plot.loc[:, "1350":"1650"].to_numpy()
    data_to_plot_2 = data_to_plot.loc[:, "1750":"2150"].to_numpy()

    print(i, set(data_to_plot['target']))

    # If lamp_power_to_plot is not None select only some lamp power to plot. Otherwise plot all the lamp power
    if lamp_power_to_plot is not None :
        lamp_idx = np.zeros(len(lamp_power_list))
        for i in range(len(lamp_power_to_plot)):
            lamp_idx = np.logical_or(lamp_idx, lamp_power_list == lamp_power_to_plot[i])
    else:
        lamp_idx = np.ones(len(lamp_power_list)) == 1

        # if i == 3:   row_to_skip = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # elif i == 4: row_to_skip = []


    fig, axs = plt.subplots(1, 2, figsize = (16, 8))
    
    axs[0].plot(wavelengts_1, data_to_plot_1[lamp_idx, :].T, label = lamp_power_list[lamp_idx])
    axs[1].plot(wavelengts_2, data_to_plot_2[lamp_idx, :].T, label = lamp_power_list[lamp_idx])

    for ax in axs: 
        ax.legend()
        ax.set_title(data_type + " - gain " + str(gain))
        ax.set_xlabel("Wavelength [nm]")
        ax.grid(True)

    fig.tight_layout()
    fig.show()
