"""
Compute the average spectra for a specific and plot it (plus std) for every time point
"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import numpy as np
import matplotlib.pyplot as plt
import os

from library import preprocess, manage_data_beans

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

plant_to_examine = 'PhaseolusVulgaris'
plant_to_examine = 'ViciaFaba'

lamp_power_to_plot = 60

plant_group = 'test_300'

use_standardization = False
compute_absorbance = True
use_sg_preprocess = True

# Parameter Savitky Golay
w = 50
p = 3
deriv = 1

plot_config = dict(
    figsize = (24, 12),
    fontsize = 15,
    save_fig = True
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

t_list = [0, 1, 2, 3, 4, 5, 6]
color_per_time_point = { 0 : 'red', 1 : 'blue', 2 : 'green', 3 : 'purple', 4 : 'orange', 5 : 'violet', 6 : 'brown'} 

plt.rcParams.update({'font.size': plot_config['fontsize']})
fig, axs = plt.subplots(1, 2, figsize = plot_config['figsize'])

for i in range(len(t_list)):
    # Get NIRS data
    path_beans = "data/beans/t{}/csv/beans.csv".format(t_list[i])
    data_beans_full, wavelength, group_labels_list, plant_labels_list, plant_type_list = manage_data_beans.read_data_beans_single_file(path_beans, return_numpy = False)
    data_beans = data_beans_full[data_beans_full['plant'] == plant_to_examine]


    # Remove measure with 0 gain (it is a single measure per plant)
    data_beans = data_beans[data_beans['gain_0'] != 0]

    # Select lamp power
    data_beans = data_beans[data_beans['lamp_0'] == lamp_power_to_plot]

    if compute_absorbance : data_beans = preprocess.R_A(data_beans) 
    if use_standardization : data_beans = preprocess.normalize_standardization(data_beans, divide_mems = True)
    if use_sg_preprocess : data_beans = preprocess.sg(data_beans, w = w, p = p, deriv = deriv)

    # Get the specific spectra and plot
    data_beans_group = data_beans[data_beans['test_control'] == plant_group]
    if len(data_beans) > 0:
        # Get the data of the two mems
        data_beans_mems_1 = data_beans_group .loc[:, "1350":"1650"]
        data_beans_mems_2 = data_beans_group.loc[:, "1750":"2150"]

        mean_spectra_mems_1 = data_beans_mems_1.mean(0).to_numpy()
        std_spectra_mems_1 = data_beans_mems_1.std(0).to_numpy()

        mean_spectra_mems_2 = data_beans_mems_2.mean(0).to_numpy()
        std_spectra_mems_2 = data_beans_mems_2.std(0).to_numpy()


        # Plot for mems 1
        axs[0].plot(np.arange(1350, 1650 + 1), mean_spectra_mems_1, 
                    label = "t{}".format(t_list[i]), color = color_per_time_point[t_list[i]],
                    )
        axs[0].fill_between(np.arange(1350, 1650 + 1), mean_spectra_mems_1 + std_spectra_mems_1, mean_spectra_mems_1 - std_spectra_mems_1,
                            color = color_per_time_point[t_list[i]], alpha = 0.25
                            )

        # Plot for mems 2 
        axs[1].plot(np.arange(1750, 2150 + 1), mean_spectra_mems_2, 
                    label = "t{}".format(t_list[i]), color = color_per_time_point[t_list[i]],
                    )
        axs[1].fill_between(np.arange(1750, 2150 + 1), mean_spectra_mems_2 + std_spectra_mems_2, mean_spectra_mems_2 - std_spectra_mems_2,
                            color = color_per_time_point[t_list[i]], alpha = 0.25
                            )
        # Plots title
        axs[0].set_title("Mems1")
        axs[1].set_title("Mems2")
    else :
        print("No plants for group {} at t{}".format(plant_group, i))

for ax in axs :
    ax.grid(True)
    ax.set_xlabel("wavelength [nm]")
    ax.legend()

fig.suptitle('Average spectra per group - {} lamp {}'.format(plant_group, lamp_power_to_plot))
fig.tight_layout()
fig.show()


if plot_config['save_fig']:
    path_save = 'Saved Results/beans_spectra/'
    os.makedirs(path_save, exist_ok = True)

    path_save += '{}_average_spectra_per_group_{}_lamp_{}'.format(plant_to_examine, plant_group, lamp_power_to_plot)
    fig.savefig(path_save + ".png", format = 'png')
    # fig.savefig(path_save + ".pdf", format = 'pdf')
