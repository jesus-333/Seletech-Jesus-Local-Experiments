"""
Plot the average for each source after extraction. Separate plot.
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

fontsize = 18

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
    
    if 'orange' in labels_list :
        update_label_orange = {'orange' : 'Orange Layer', 
                               'white' : 'White Layer',
                               'orangeDOWN_whiteUP' : 'Two Layer', 
                               'whole_orange' : 'Whole Orange' }
        for i in range(len(labels_list)):
            labels_list[i] = update_label_orange[labels_list[i]]

    # (Optional) Apply minmax scaling
    if use_minmax : 
        scaler = MinMaxScaler()
        data.loc[:, "1350":"2150"] = scaler.fit_transform(data.loc[:, "1350":"2150"])
    
    # Create figure
    
    # Plot average andd std for each label
    figsize = (12, 8)
    fig_1, ax_1 = plt.subplots(1, 1, figsize = figsize)
    fig_2, ax_2 = plt.subplots(1, 1, figsize = figsize)
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
        ax_1.plot(wavelength[idx_mems_1], avg_mems_1, label = label)
        ax_1.fill_between(wavelength[idx_mems_1], avg_mems_1 - std_mems_1, avg_mems_1 + std_mems_1, alpha = 0.2)
        ax_1.set_xlim([wavelength[idx_mems_1][0], wavelength[idx_mems_1][-1]])

        # Plot mems 2
        ax_2.plot(wavelength[idx_mems_2], avg_mems_2, label = label)
        ax_2.fill_between(wavelength[idx_mems_2], avg_mems_2 - std_mems_2, avg_mems_2 + std_mems_2, alpha = 0.2)
        ax_2.set_xlim([wavelength[idx_mems_2][0], wavelength[idx_mems_2][-1]])

        tmp_ax_list = [ax_1, ax_2]
        tmp_fig_list = [fig_1, fig_2]

        for i in range(2) :
            ax = tmp_ax_list[i]
            fig = tmp_fig_list[i]

            ax.set_xlabel("Wavelength", fontsize = fontsize)
            ax.legend(fontsize = 16)
            # ax.set_title(title, fontsize = fontsize)
            ax.grid(True)
            ax.tick_params(axis='both', which='major', labelsize = fontsize)

            fig.tight_layout()
            fig.show()

            path_save = "Saved Results/merged_dataset/paper/"
            os.makedirs(path_save, exist_ok = True)
            fig.savefig(path_save + title + "_MEMS_{}.png".format(i + 1))
            fig.savefig(path_save + title + "_MEMS_{}.pdf".format(i + 1))

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
