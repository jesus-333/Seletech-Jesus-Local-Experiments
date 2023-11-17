import numpy as np
import matplotlib.pyplot as plt

from library import preprocess, manage_data_beans

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

plant_to_examine = 'PhaseolusVulgaris'
plant_to_examine = 'ViciaFaba'

idx_spectra = 2

lamp_power_spectra_to_plot = 60
gain_spectra_to_plot = 1
group_spectra_to_plot = 'test_300' # Possible value are control, test_150, test_300

t_list = [0, 1, 2, 3, 4, 5, 6]

use_standardization = False
use_control_group_to_calibrate = True
norm_type_with_control_group = 2 # Used only if use_control_group_to_calibrate == True
use_sg_preprocess = False

normalize_per_lamp_power = True # If true normalize each group of lamp power separately

plot_config = dict(
    figsize = (24, 12),
    fontsize = 15,
    save_fig = True
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

plt.rcParams.update({'font.size': plot_config['fontsize']})
fig, axs = plt.subplots(1, 2, figsize = plot_config['figsize'])

# Used to get the always the same plant inside the group
if group_spectra_to_plot == 'control' : n_plant = 'CON1'
if group_spectra_to_plot == 'test_150' : n_plant = 'NACL150_1'
if group_spectra_to_plot == 'test_300' : n_plant = 'NACL300_2'

for i in range(len(t_list)):
    # Get NIRS data
    path_beans = "data/beans/t{}/csv/beans.csv".format(t_list[i])
    data_beans_full, wavelength, group_labels_list, plant_labels_list, plant_type_list = manage_data_beans.read_data_beans_single_file(path_beans, return_numpy = False)
    data_beans = data_beans_full[data_beans_full['plant'] == plant_to_examine]

    # Normalization
    if normalize_per_lamp_power :
        lamp_power_list = [50, 60, 70, 80]
        for lamp_power in lamp_power_list :
            data_lamp_power = data_beans[data_beans['lamp_0'] == lamp_power]

            if use_standardization  : data_lamp_power = preprocess.normalize_standardization(data_lamp_power, divide_mems = True)
            if use_control_group_to_calibrate : data_lamp_power = preprocess.normalize_with_control_group(data_lamp_power, norm_type = norm_type_with_control_group)
            if use_sg_preprocess : data_lamp_power = preprocess.sg(data_lamp_power)

            data_beans[data_beans['lamp_0'] == lamp_power] = data_lamp_power

    else :
        if use_standardization  : data_beans = preprocess.normalize_standardization(data_beans, divide_mems = True)
        if use_control_group_to_calibrate : data_beans = preprocess.normalize_with_control_group(data_beans, norm_type = norm_type_with_control_group)
        if use_sg_preprocess : data_beans = preprocess.sg(data_beans)

    
    # Get the specific spectra and plot
    data_beans = data_beans[data_beans['test_control'] == group_spectra_to_plot]
    if len(data_beans) > 0:
        data_beans = data_beans[data_beans['gain_0'] == gain_spectra_to_plot]
        data_beans = data_beans[data_beans['lamp_0'] == lamp_power_spectra_to_plot]
        data_beans = data_beans[data_beans['type'] == n_plant]
        spectra_to_plot_mems_1 = data_beans.loc[:, "1350":"1650"].to_numpy()[idx_spectra]
        spectra_to_plot_mems_2 = data_beans.loc[:, "1750":"2150"].to_numpy()[idx_spectra]

        axs[0].plot(np.arange(1350, 1650 + 1), spectra_to_plot_mems_1, label = 't{}'.format(t_list[i]))
        axs[1].plot(np.arange(1750, 2150 + 1), spectra_to_plot_mems_2, label = 't{}'.format(t_list[i]))

        axs[0].set_title("Mems1")
        axs[1].set_title("Mems2")

for ax in axs :
    ax.grid(True)
    ax.set_xlabel("wavelength [nm]")
    ax.legend()

fig.suptitle(n_plant)
fig.tight_layout()
fig.show()
