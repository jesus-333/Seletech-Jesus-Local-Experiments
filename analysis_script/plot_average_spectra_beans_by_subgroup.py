import matplotlib.pyplot as plt
import os

from library import manage_data_beans, preprocess, config

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

t_list = [0, 1, 2, 3, 4, 5, 6] # Different day (see path)

# plant = 'PhaseolusVulgaris'
plant = 'ViciaFaba'

plot_config = dict(
    figsize = (100, 25),
    fontsize = 15,
    save_fig = True
)

preprocess_config = config.get_config_preprocess()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

plant_type_phaseoulus = ['CON1', 'CON2', 'CON3', 'CON4', 'CON5', 'NACL150_2', 'NACL150_3', 'NACL150_4', 'NACL150_5']
plant_labels_to_axs = {'CON1' : 0, 'CON2' : 0, 'CON3' : 0, 'CON4' : 0, 'CON5' : 0, 
                      'NACL150_1' : 1, 'NACL150_2' : 1, 'NACL150_3' : 1, 'NACL150_4' : 1, 'NACL150_5' : 1,
                      'NACL300_1' : 2, 'NACL300_2' : 2, 'NACL300_3' : 2, 'NACL300_4' : 2, 'NACL300_5' : 2
                      }

idx_to_color_dict = {0 : 'red', 1 : 'orange', 2 : 'blue', 3 : 'darkviolet', 4 : 'green'}
plant_labels_to_line = {'CON1' : 0, 'CON2' : 1, 'CON3' : 2, 'CON4' : 3, 'CON5' : 4, 
                      'NACL150_1' : 0, 'NACL150_2' : 1, 'NACL150_3' : 2, 'NACL150_4' : 3, 'NACL150_5' : 4,
                      'NACL300_1' : 0, 'NACL300_2' : 1, 'NACL300_3' : 2, 'NACL300_4' : 3, 'NACL300_5' : 4
                      }


plt.rcParams.update({'font.size': plot_config['fontsize']})
fig, axs = plt.subplots(3, len(t_list) , figsize = plot_config['figsize'])

for i in range(len(t_list)):
    path = "data/beans/t{}/csv/beans.csv".format(t_list[i])

    data, wavelength, group_labels_list, plant_labels_list, plant_type_list = manage_data_beans.read_data_beans_single_file(path, return_numpy = False)

    data = data[data['plant'] == plant]

    data = preprocess.beans_preprocess_pipeline(data, preprocess_config, wavelength)

    mean_dict, std_dict = manage_data_beans.compute_average_and_std_per_subgroup(data, plant_labels_list)

    for j in range(len(plant_labels_list)):
        plant_label = plant_labels_list[j]

        idx_axs = plant_labels_to_axs[plant_label]
        idx_line = plant_labels_to_line[plant_label]
        
        print(plant_labels_list[j], idx_axs)

        ax = axs[idx_axs, i]
        ax.plot(wavelength, mean_dict[plant_label], 
                label = plant_label, color = idx_to_color_dict[idx_line], linewidth = 2)
        # ax.fill_between(wavelength, mean_dict[plant_label] + std_dict[plant_label], mean_dict[plant_label] - std_dict[plant_label], 
        #                 alpha = 0.25, color = idx_to_color_dict[idx_line])
        # ax.set_ylim([750, 2750])
        ax.set_xlim([1350, 2150])
        ax.set_title('{} - t{}'.format(plant_label, t_list[i]))
        ax.grid(True)
        ax.legend()
    print("- - - - - - - - - - - - ")

fig.tight_layout()
fig.show()


if plot_config['save_fig']:
    path_save = 'Saved Results/beans_spectra/'
    os.makedirs(path_save, exist_ok = True)

    path_save = 'Saved Results/beans_spectra/'
    path_save += 'average_spectra_different_group_{}'.format(plant)
    fig.savefig(path_save + ".png", format = 'png')
    fig.savefig(path_save + ".pdf", format = 'pdf')
