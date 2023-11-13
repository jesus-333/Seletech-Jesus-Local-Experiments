"""
For the beans experiment plot across time the 1450nm and the 1950nm wavelength
For each group (control, test_150, test_300) visualize the average value of the lenght across time, average among plants.
E.g. if a group has 3 plants and for each plant there are 5 measure the average is taken over those 5 measurement.Eaech group has a different plot.

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
# plant_to_examine = 'ViciaFaba'
t_list = [0, 1, 2, 3, 4, 5, 6]

use_sg_preprocess = False

plot_config = dict(
    figsize = (12, 8),
    fontsize = 15,
    save_fig = True
)

use_shaded_area = False

wavelength_to_plot = "1450"
# wavelength_to_plot = "1950"

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

if plant_to_examine == 'PhaseolusVulgaris': plant_group_list = ['control', 'test_150',]
else : plant_group_list = ['control', 'test_150', 'test_300']
color_per_group = {'control' : 'blue', 'test_150' : 'green', 'test_300' : 'red'}

for group in plant_group_list:
    print(group)

    wavelength_mean_per_plant = dict()
    wavelength_std_per_plant = dict()

    for i in range(len(t_list)):
        # Get NIRS data 
        path_beans = "data/beans/t{}/csv/beans.csv".format(t_list[i])
        data_beans_full, wavelength, _, _, _ = manage_data_beans.read_data_beans_single_file(path_beans, return_numpy = False)

        # Seletct plant, group
        data_beans = data_beans_full[data_beans_full['plant'] == plant_to_examine]
        data_beans = data_beans[data_beans['test_control'] == group]

        # Remove measure with 0 gains (it is a single measure per plant)
        data_beans = data_beans[data_beans['gain_0'] != 0]
        
        # Preprocess and average per plant (CON1, CON2 etc)(the field it is called type inside the csv)
        if use_sg_preprocess : data_beans = preprocess.sg(data_beans)
        grouped_data = data_beans.groupby('type')

        # Get wavelength
        wavelength_mean = grouped_data[wavelength_to_plot].mean()
        wavelength_std = grouped_data[wavelength_to_plot].std()

        # Get list of plants inside the group
        plant_labels_list = list(set(data_beans['type']))
        plant_labels_list.sort()
        if i == 0: labels_to_plot = plant_labels_list.copy()

        for j in range(len(labels_to_plot)): 
            plant_label = labels_to_plot[j]
            
            if i == 0: # Only during the first day create empty list to save the data
                wavelength_std_per_plant[plant_label] = []
                wavelength_mean_per_plant[plant_label] = []

            if plant_label in wavelength_mean:
                wavelength_std_per_plant[plant_label].append(wavelength_std[plant_label])
                wavelength_mean_per_plant[plant_label].append(wavelength_mean[plant_label])
            else:
                wavelength_mean_per_plant[plant_label].append(wavelength_mean_per_plant[plant_label][-1])
                wavelength_std_per_plant[plant_label].append(wavelength_std_per_plant[plant_label][-1])

    plt.rcParams.update({'font.size': plot_config['fontsize']})
    fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])
    for plant_label in labels_to_plot:
        if use_shaded_area:
            ax.plot(t_list, wavelength_mean_per_plant[plant_label],
                        label = plant_label, marker = 'o'
                        )
            ax.fill_between(t_list, np.asarray( wavelength_mean_per_plant[plant_label] ) + np.asarray( wavelength_std_per_plant[plant_label] ),
                            np.asarray( wavelength_mean_per_plant[plant_label] ) - np.asarray( wavelength_std_per_plant[plant_label] ),
                            alpha = 0.25, 
                            )
        else:
            ax.errorbar(t_list, wavelength_mean_per_plant[plant_label], wavelength_std_per_plant[plant_label],
                        label = plant_label, marker = 'o', capsize = 8,
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
        path_save += '{}_{}_wavelength_{}_per_plant_beans'.format(group, plant_to_examine, wavelength_to_plot)
        fig.savefig(path_save + ".png", format = 'png')
        # fig.savefig(path_save + ".pdf", format = 'pdf')
