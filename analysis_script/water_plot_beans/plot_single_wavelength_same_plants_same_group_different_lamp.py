"""
For the beans experiment plot across time the 1450nm and the 1950nm wavelength
Select a group (control, test_150, test_300)  visualize the average value of the wavelenght across time, average among the different lamp power.

@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

import numpy as np
import matplotlib.pyplot as plt
import os

from library import manage_data_other_sensors, manage_data_beans, timestamp_functions
from library import preprocess

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Settings 

plant_to_examine = 'PhaseolusVulgaris'
# plant_to_examine = 'ViciaFba'
t_list = [0, 1, 2, 3, 4, 5, 6]

use_sg_preprocess = False

plot_config = dict(
    figsize = (12, 8),
    fontsize = 15,
    save_fig = True
)

use_shaded_area = False

wavelength_to_plot = "1450"
wavelength_to_plot = "1950"

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

lamp_power_list = [50, 60, 70, 80]

if plant_to_examine == 'PhaseolusVulgaris': plant_group_list = ['control', 'test_150',]
else : plant_group_list = ['control', 'test_150', 'test_300']
color_per_group = {'control' : 'blue', 'test_150' : 'green', 'test_300' : 'red'}

for group in plant_group_list:
    print(group)

    wavelength_mean_per_lamp_power = dict()
    wavelength_std_per_lamp_power = dict()

    for i in range(len(t_list)):
        # Get NIRS data 
        path_beans = "data/beans/t{}/csv/beans.csv".format(t_list[i])
        data_beans_full, wavelength, _, _, _ = manage_data_beans.read_data_beans_single_file(path_beans, return_numpy = False)

        # Seletct plant, group
        data_beans = data_beans_full[data_beans_full['plant'] == plant_to_examine]
        data_beans = data_beans[data_beans['test_control'] == group]

        # Remove measure with 0 gain (it is a single measure per plant)
        data_beans = data_beans[data_beans['gain_0'] != 0]
        
        # Preprocess and average per lamp power 
        if use_sg_preprocess : data_beans = preprocess.sg(data_beans)
        grouped_data = data_beans.groupby('lamp_0')

        # Get wavelength
        wavelength_mean = grouped_data[wavelength_to_plot].mean()
        wavelength_std = grouped_data[wavelength_to_plot].std()

        for j in range(len(lamp_power_list)): 
            lamp_power = lamp_power_list[j]
            
            if i == 0: # Only during the first day create empty list to save the data
                wavelength_std_per_lamp_power[lamp_power] = []
                wavelength_mean_per_lamp_power[lamp_power] = []

            if lamp_power in wavelength_mean:
                wavelength_std_per_lamp_power[lamp_power].append(wavelength_std[lamp_power])
                wavelength_mean_per_lamp_power[lamp_power].append(wavelength_mean[lamp_power])
            else:
                wavelength_mean_per_lamp_power[lamp_power].append(wavelength_mean_per_lamp_power[lamp_power][-1])
                wavelength_std_per_lamp_power[lamp_power].append(wavelength_std_per_lamp_power[lamp_power][-1])

    plt.rcParams.update({'font.size': plot_config['fontsize']})
    fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])
    for lamp_power in lamp_power_list:
        if use_shaded_area:
            ax.plot(t_list, wavelength_mean_per_lamp_power[lamp_power],
                        label = lamp_power, marker = 'o'
                        )
            ax.fill_between(t_list, np.asarray( wavelength_mean_per_lamp_power[lamp_power] ) + np.asarray( wavelength_std_per_lamp_power[lamp_power] ),
                            np.asarray( wavelength_mean_per_lamp_power[lamp_power] ) - np.asarray( wavelength_std_per_lamp_power[lamp_power] ),
                            alpha = 0.25, 
                            )
        else:
            ax.errorbar(t_list, wavelength_mean_per_lamp_power[lamp_power], wavelength_std_per_lamp_power[lamp_power],
                        label = lamp_power, marker = 'o', capsize = 8,
                        )

    ax.set_xlabel("Time point")
    ax.set_ylabel("Amplitude")
    ax.set_title("Wavelength {} evolution - {} {}".format(wavelength_to_plot, plant_to_examine, group))
    ax.legend()
    ax.grid()

    fig.tight_layout()
    fig.show()

    if plot_config['save_fig']:
        path_save = 'Saved Results/beans_spectra/'
        os.makedirs(path_save, exist_ok = True)

        path_save = 'Saved Results/beans_spectra/'
        path_save += '{}_{}_wavelength_{}_per_lamp_power_beans'.format(group, plant_to_examine, wavelength_to_plot)
        fig.savefig(path_save + ".png", format = 'png')
        # fig.savefig(path_save + ".pdf", format = 'pdf')
