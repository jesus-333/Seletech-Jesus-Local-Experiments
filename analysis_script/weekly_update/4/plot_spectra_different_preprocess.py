# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import pandas as pd
import matplotlib.pyplot as plt
import os

from library import manage_data_beans, preprocess

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

plant = 'PhaseolusVulgaris'
# plant = 'ViciaFaba'

t = 0
lamp_power = 80
group = 'test_150'

# Parameter to remove spectra with amplitude to low
min_amplitude = 1000
percentage_above_threshold = 0.8

# Parameter for preprocess
w_list = [20, 30, 40, 50, 35]
w_list = [10]
p_list = [3]
deriv_list = [2]

mems_to_plot = 1

plot_config = dict(
    figsize = (12, 8),
    fontsize = 20,
    add_std = True,
    save_fig = True,
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

plt.rcParams.update({'font.size': plot_config['fontsize']})

lamp_to_color = {50 : 'red', 60 : 'black', 70 : 'green', 80 : 'orange'}
group_color = {'test_300' : 'red', 'control' : 'green', 'test_150' : 'blue'}
group_linestyle = {'test_300' : 'dashed', 'control' : 'solid', 'test_150' : 'dashdot'}
group_linewidth = {'test_300' : 4, 'control' : 1, 'test_150' : 2}

# Create figure for the plots
fig, axs = plt.subplots(1, 1, figsize = plot_config['figsize'])

# Load data
path_spectra = "data/beans/t{}/csv/beans_avg.csv".format(t)
spectra_data_ALL, wavelength, group_labels_list, plant_labels_list, plant_type_list = manage_data_beans.read_data_beans_single_file(path_spectra, return_numpy = False)
spectra_data_ALL = spectra_data_ALL[spectra_data_ALL['gain_0'] == 1]
if plant is not None : spectra_data_ALL  = spectra_data_ALL[spectra_data_ALL['plant'] == plant]

# Remove spectra with at least a portion of spectra below the threshold
spectra_data_ALL , idx_data_to_keep = preprocess.filter_spectra_by_threshold(spectra_data_ALL , threshold = min_amplitude, percentage_above_threshold = percentage_above_threshold)

for w in w_list :
    for p in p_list :
        for deriv in deriv_list :

            # Preprocess
            meta_data = spectra_data_ALL .loc[:, "timestamp":"type"]
            spectra_data = preprocess.R_A(spectra_data_ALL , keep_meta = False)
            spectra_data = preprocess.normalize_standardization(spectra_data, divide_mems = True)
            spectra_data = preprocess.sg(spectra_data, w, p, deriv, keep_meta = False)
            spectra_data = pd.concat([meta_data, spectra_data], axis = 1)
            
            # Get spectra with specific lamp power
            spectra_data = spectra_data[spectra_data['test_control'] == group]
            tmp_spectra = spectra_data[spectra_data['lamp_0'] == lamp_power]

            # Compute mean and std
            tmp_spectra_mean = tmp_spectra.mean(numeric_only = True)
            tmp_spectra_std = tmp_spectra.std(numeric_only = True)

            if len(tmp_spectra) > 0:
                # Select mems to plot
                if mems_to_plot == 1 :
                    tmp_spectra_mean = tmp_spectra_mean.loc["1350":"1650"]
                    tmp_spectra_std = tmp_spectra_std.loc["1350":"1650"]
                    tmp_wavelength = wavelength[wavelength <= 1650]
                elif mems_to_plot == 2 :
                    tmp_spectra_mean = tmp_spectra_mean.loc["1750":"2150"]
                    tmp_spectra_std = tmp_spectra_std.loc["1750":"2150"]
                    tmp_wavelength = wavelength[wavelength >= 1750]
                elif mems_to_plot == 'both' :
                    tmp_spectra_mean = tmp_spectra_mean.loc[:, "1350":"2150"]
                    tmp_spectra_std = tmp_spectra_std.loc[:, "1350":"2150"]
                    tmp_wavelength = wavelength[:]
                else:
                    raise ValueError("mems_to_plot must have value 1 or 2 or both")
                
                fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])
                
                ax.plot(tmp_wavelength, tmp_spectra_mean, color = group_color[group])

                ax.fill_between(tmp_wavelength, tmp_spectra_mean - tmp_spectra_std, tmp_spectra_mean + tmp_spectra_std,
                                color = group_color[group], linestyle = group_linestyle[group], linewidth = group_linewidth[group],
                                alpha = 0.25
                                )

                ax.grid(True)

                ax.set_ylabel("Amplitude")
                ax.set_xlabel("Wavelength [nm]")

                # ax.set_xlim([1370, 1625])
                # if plant == 'ViciaFaba' : 
                #     ax.set_ylim([-0.0007, 0.00025])
                # else : 
                #     ax.set_ylim([-0.0007, 0.00025])

                ax.set_title("w = {}, p = {}, der = {}".format(w, p, deriv))

                fig.tight_layout()
                fig.show()

                if plot_config['save_fig'] :
                    path_save = 'Saved Results/weekly_update_beans/4/preprocess/'
                    os.makedirs(path_save, exist_ok = True)

                    path_save += '{}_{}_lamp_{}_w_{}_p_{}_der_{}_mems_{}'.format(plant, group, lamp_power, w, p, deriv, mems_to_plot)
                    fig.savefig(path_save + ".png", format = 'png')
