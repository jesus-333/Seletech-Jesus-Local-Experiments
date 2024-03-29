"""
Script per dei risultati per il 4 meeting.
Crea degli istogrammi dei samples.

In questo script viene creato una figura per ogni giorno. In ogni giorno ci sono 4 plot, 1 per lamp power, dove vengono visualizzate media e std dei gruppi di piante
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

normalize_hist = True
mems_to_plot = 2

plot_config = dict(
    figsize = (20, 12),
    n_bins = 50,
    fontsize = 20,
    linewidth = 2,
    save_fig = True,
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

plt.rcParams.update({'font.size': plot_config['fontsize']})

lamp_power_list = [50, 60, 70, 80]

for i in range(len(t_list)):
    t = t_list[i]

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
    
    # Create figure for the plots
    fig, axs = plt.subplots(2, 2, figsize = plot_config['figsize'])
    
    idx_lamp_power = 0
    for j in range(2):
        for k in range(2):
            # Get spectra with specific lamp power
            lamp_power = lamp_power_list[idx_lamp_power]
            idx_lamp_power += 1
            tmp_spectra = spectra_data[spectra_data['lamp_0'] == lamp_power]

            # Select axis for the plot
            ax = axs[j, k]

            if len(tmp_spectra) > 0:
                
                # Plot the spectra for each group
                for plant_group in set(tmp_spectra['test_control']):

                    spectra_to_plot = tmp_spectra[tmp_spectra['test_control'] == plant_group]

                    # Select mems to plot
                    if mems_to_plot == 1 :
                        spectra_to_plot = spectra_to_plot.loc[:, "1350":"1650"]
                    elif mems_to_plot == 2 :
                        spectra_to_plot = spectra_to_plot.loc[:, "1750":"2150"]
                    elif mems_to_plot == 'both' :
                        spectra_to_plot = spectra_to_plot.loc[:, "1350":"2150"]
                    else:
                        raise ValueError("mems_to_plot must have value 1 or 2 or both")

                    # Convert to numpy, remove border artifacts and flat the data
                    spectra_to_plot = spectra_to_plot.to_numpy()
                    spectra_to_plot = spectra_to_plot[:, int(w / 2):-int(w / 2)].flatten()
                    
                    # Plot the average spectra per group
                    ax.hist(spectra_to_plot, plot_config['n_bins'], density = normalize_hist,
                            histtype = 'step', label = plant_group, linewidth = plot_config['linewidth']
                            )

                # Add info to the plot
                ax.legend()
                ax.grid(True)
                ax.set_xlabel("Amplitude")
                if mems_to_plot == 1:
                    ax.set_xlim([-1 * 1e-5, 0.5 * 1e-5])
                elif mems_to_plot == 2:
                    ax.set_xlim([-1.1 * 1e-5, 1.1 * 1e-5])
                if normalize_hist : ax.set_ylim([0, 400000])

            ax.set_title("Lamp power {}".format(lamp_power))
    
    fig.suptitle("{} - Day t = {}".format(plant, t))
    fig.tight_layout()
    fig.show()

    if plot_config['save_fig'] :
        path_save = 'Saved Results/weekly_update_beans/3/V1_mems_{}/'.format(mems_to_plot)
        os.makedirs(path_save, exist_ok = True)

        path_save += '3_hist_V1_t_{}_w_{}_p_{}_der_{}_mems_{}'.format(t, w, p, deriv, mems_to_plot)
        fig.savefig(path_save + ".png", format = 'png')
