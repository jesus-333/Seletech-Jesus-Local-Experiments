"""
Plot the average for each source after extraction
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings

use_minmax = False

path_beans = "data/merged_dataset/no_minmax_used_for_plot/beans.csv"
path_orange = "data/merged_dataset/no_minmax_used_for_plot/orange.csv"
path_potos = "data/merged_dataset/no_minmax_used_for_plot/potos.csv"

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Read data

data_beans  = pd.read_csv(path_beans, index_col = 0)
data_orange = pd.read_csv(path_orange, index_col = 0)
data_potos  = pd.read_csv(path_potos, index_col = 0)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def plot_average_and_std(data, title, use_minmax = False):
    """
    Plot the average for each source after extraction
    """
    
    # Get wavelength array
    starting_wavelength_dataframe = int(data.columns[0])
    w = (starting_wavelength_dataframe - 1350) * 2
    tmp_wave_1_mems_1 = int(1350 + w / 2)
    tmp_wave_2_mems_1 = int(1650 - w / 2)
    tmp_wave_1_mems_2 = int(1750 + w / 2)
    tmp_wave_2_mems_2 = int(2150 - w / 2)
    wavelength = np.hstack((np.arange(tmp_wave_1_mems_1, tmp_wave_2_mems_1 + 1), np.arange(tmp_wave_1_mems_2, tmp_wave_2_mems_2 + 1)))
    
    # Get label
    labels_list = list(set(data["label_text"]))

    # (Optional) Apply minmax scaling
    if use_minmax : 
        scaler = MinMaxScaler()
        data.loc[:, "1350":"2150"] = scaler.fit_transform(data.loc[:, "1350":"2150"])
    
    # Create figure
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot average andd std for each label
    for label in labels_list:
        print(label)
        idx = data["label_text"] == label
        tmp_data = data.loc[:, "1350":"2150"].to_numpy()
        tmp_data = tmp_data[idx]
        
        # Compute indices for each mems
        idx_mems_1 = wavelength <= tmp_wave_2_mems_1
        idx_mems_2 = wavelength >= tmp_wave_1_mems_2

        # Get data for each mems
        data_mems_1 = tmp_data[:, idx_mems_1]
        data_mems_2 = tmp_data[:, idx_mems_2]
        
        # Compute average and std
        avg_mems_1 = np.mean(data_mems_1, axis = 0)
        avg_mems_2 = np.mean(data_mems_2, axis = 0)
        std_mems_1 = np.std(data_mems_1, axis = 0)
        std_mems_2 = np.std(data_mems_2, axis = 0)

        # Plot mems 1
        axs[0].plot(wavelength[idx_mems_1], avg_mems_1, label = label)
        axs[0].fill_between(wavelength[idx_mems_1], avg_mems_1 - std_mems_1, avg_mems_1 + std_mems_1, alpha = 0.2)

        # Plot mems 2
        axs[1].plot(wavelength[idx_mems_2], avg_mems_2, label = label)
        axs[1].fill_between(wavelength[idx_mems_2], avg_mems_2 - std_mems_2, avg_mems_2 + std_mems_2, alpha = 0.2)

        for ax in axs :
            ax.set_xlabel("Wavelength")
            ax.legend()
            ax.set_title(title)
            ax.grid(True)

    fig.tight_layout()
    fig.show()

    path_save = "Saved Results/merged_dataset/"
    os.makedirs(path_save, exist_ok = True)
    fig.savefig(path_save + title + ".png")

plot_average_and_std(data_beans, "Beans", use_minmax)
plot_average_and_std(data_orange, "Orange", use_minmax)
plot_average_and_std(data_potos, "Potos", use_minmax)

# print(data_beans.max())
# print(data_beans.min())
# print(data_orange.max())
# print(data_orange.min())
# print(data_potos.max())
# print(data_potos.min())
# print(data_beans.mean())
# print(data_orange.mean())
# print(data_potos.mean())
