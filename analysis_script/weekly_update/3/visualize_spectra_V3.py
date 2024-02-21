"""
Script per dei risultati per il 4 meeting.
Cercare i picchi come ha suggerito Dag.

In questo script viene creata una figura per ogni lamp power. Su figura ci sono 3 plot, 1 per ogni tipo di pianta, dove vengono visualizzati media e std per i vari giorni.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import pandas as pd
import matplotlib.pyplot as plt
import os

from library import manage_data_beans, preprocess

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

t_list = [0, 1, 2, 3, 4, 5, 6]
# t_list = [0]

# plant = 'PhaseolusVulgaris'
plant = 'ViciaFaba'

# Parameter to remove spectra with amplitude to low
min_amplitude = 1000
percentage_above_threshold = 0.8

# Parameter for preprocess
compute_absorbance = True
use_sg_filter = True
w = 50
p = 3
deriv = 2

mems_to_plot = 2

plot_config = dict(
    figsize = (30, 12),
    fontsize = 20,
    add_std = True,
    save_fig = True,
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

plt.rcParams.update({'font.size': plot_config['fontsize']})

lamp_power_list = [50, 60, 70, 80]
lamp_to_color = {50 : 'red', 60 : 'black', 70 : 'green', 80 : 'orange'}

for idx_lamp_power in range(len(lamp_power_list)):

    # Create figure for the plots
    fig, axs = plt.subplots(1, 3, figsize = plot_config['figsize'])

    for idx_day in range(len(t_list)):
        t = t_list[idx_day]

        # Load data
        path_spectra = "data/beans/t{}/csv/beans_avg.csv".format(t)
        spectra_data_ALL, wavelength, group_labels_list, plant_labels_list, plant_type_list = manage_data_beans.read_data_beans_single_file(path_spectra, return_numpy = False)
        spectra_data_ALL = spectra_data_ALL[spectra_data_ALL['gain_0'] == 1]
        if plant is not None : spectra_data = spectra_data_ALL[spectra_data_ALL['plant'] == plant]

        # Remove spectra with at least a portion of spectra below the threshold
        spectra_data, idx_data_to_keep = preprocess.filter_spectra_by_threshold(spectra_data, threshold = min_amplitude, percentage_above_threshold = percentage_above_threshold)

        # Preprocess
        meta_data = spectra_data.loc[:, "timestamp":"type"]
        if compute_absorbance : spectra_data = preprocess.R_A(spectra_data, keep_meta = False)
        if use_sg_filter : spectra_data = preprocess.sg(spectra_data, w, p, deriv, keep_meta = False)
        if compute_absorbance or use_sg_filter: # Since during preprocess the metadata are removed here are restored
            spectra_data = pd.concat([meta_data, spectra_data], axis = 1)
        else:
            raise ValueError("At least 1 between compute_absorbance and use_sg_filter must be true")
        
        # Get spectra with specific lamp power
        lamp_power = lamp_power_list[idx_lamp_power]
        tmp_spectra = spectra_data[spectra_data['lamp_0'] == lamp_power]

        if len(tmp_spectra) > 0:
            # Get average and std per group
            tmp_spectra_mean = tmp_spectra.groupby("test_control").mean(numeric_only = True)
            tmp_spectra_std = tmp_spectra.groupby("test_control").std(numeric_only = True)
            
            # Select mems to plot
            if mems_to_plot == 1 :
                tmp_spectra_mean = tmp_spectra_mean.loc[:, "1350":"1650"]
                tmp_spectra_std = tmp_spectra_std.loc[:, "1350":"1650"]
                tmp_wavelength = wavelength[wavelength <= 1650]
            elif mems_to_plot == 2 :
                tmp_spectra_mean = tmp_spectra_mean.loc[:, "1750":"2150"]
                tmp_spectra_std = tmp_spectra_std.loc[:, "1750":"2150"]
                tmp_wavelength = wavelength[wavelength >= 1750]
            elif mems_to_plot == 'both' :
                tmp_spectra_mean = tmp_spectra_mean.loc[:, "1350":"2150"]
                tmp_spectra_std = tmp_spectra_std.loc[:, "1350":"2150"]
                tmp_wavelength = wavelength[:]
            else:
                raise ValueError("mems_to_plot must have value 1 or 2 or both")

            # Plot the spectra for each group
            for idx_group in range(len(tmp_spectra_mean)):
                # Select axis for the plot
                ax = axs[idx_group]

                spectra_to_plot_mean = tmp_spectra_mean.iloc[idx_group, :]
                spectra_to_plot_std = tmp_spectra_std.iloc[idx_group, :]
                
                # Plot the average spectra per group
                ax.plot(tmp_wavelength, spectra_to_plot_mean,
                        label = "t = {}".format(t),
                        )
                
                # (OPTIONAL) Add the std
                if plot_config['add_std']:
                    ax.fill_between(tmp_wavelength, spectra_to_plot_mean + spectra_to_plot_std, spectra_to_plot_mean - spectra_to_plot_std,
                                    alpha = 0.25,
                                    )

                # Add info to the plot
                ax.legend(fontsize = 15)
                ax.grid(True)
                ax.set_xlabel("Wavelength [nm]")
                ax.set_xlim([tmp_wavelength[0], tmp_wavelength[-1]])
                
                if mems_to_plot == 1:
                    # ax.set_ylim([-1 * 1e-5, 1.5 * 1e-5])
                    ax.set_ylim([-0.85 * 1e-5, 0.3 * 1e-5])
                elif mems_to_plot == 2:
                    ax.set_ylim([-1.1 * 1e-5, 1.1 * 1e-5])

                ax.set_title(spectra_to_plot_mean.name)
        
    fig.suptitle("{} - lamp power {}".format(plant, lamp_power))
    fig.tight_layout()
    fig.show()

    if plot_config['save_fig'] :
        path_save = 'Saved Results/weekly_update_beans/3/V3_mems_{}/'.format(mems_to_plot)
        os.makedirs(path_save, exist_ok = True)

        path_save += '3_peak_V3_lamp_{}_w_{}_p_{}_der_{}_mems_{}'.format(lamp_power, w, p, deriv, mems_to_plot)
        fig.savefig(path_save + ".png", format = 'png')
