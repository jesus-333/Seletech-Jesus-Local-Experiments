import matplotlib.pyplot as plt
import os

from library import manage_data_beans

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

t_list = [0, 1, 2, 3, 4, 5, 6] # Different day (see path)

plant = 'PhaseolusVulgaris'
# plant = 'ViciaFaba's

plot_config = dict(
    figsize = (25, 40),
    fontsize = 15,
    save_fig = True
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

plant_type_phaseoulus = ['CON1', 'CON2', 'CON3', 'CON4', 'CON5', 'NACL150_2', 'NACL150_3', 'NACL150_4', 'NACL150_5']
plant_labels_to_dict = {'CON1' : 0, 'CON2' : 0, 'CON3' : 0, 'CON4' : 0, 'CON5' : 0, 
                      'NACL150_1' : 1, 'NACL150_2' : 1, 'NACL150_3' : 1, 'NACL150_4' : 1, 'NACL150_5' : 1,
                      'NACL300_1' : 2, 'NACL300_2' : 2, 'NACL300_3' : 2, 'NACL300_4' : 2, 'NACL300_5' : 2
                      }


plt.rcParams.update({'font.size': plot_config['fontsize']})
fig, axs = plt.subplots(len(t_list), 3, figsize = plot_config['figsize'])

for i in range(len(t_list)):
    path = "data/beans/t{}/csv/beans.csv".format(t_list[i])

    data, wavelength, group_labels_list, plant_labels_list, plant_type_list = manage_data_beans.read_data_beans_single_file(path, return_numpy = False)

    data = data[data['plant'] == plant]

    mean_dict, std_dict = manage_data_beans.compute_average_and_std_per_subgroup(data, plant_labels_list)

    for j in range(len(plant_labels_list)):
        plant_label = plant_labels_list[j]

        idx = plant_labels_to_dict[plant_label]
        print(plant_labels_list[j], idx)

        ax = axs[i, idx]
        ax.plot(wavelength, mean_dict[plant_label], label = plant_label)
        ax.fill_between(wavelength, mean_dict[plant_label] + std_dict[plant_label], mean_dict[plant_label] - std_dict[plant_label], alpha = 0.25)
        ax.set_ylim([750, 2750])
        ax.set_xlim([1350, 2150])
        ax.set_title('{} - t{}'.format(plant_label, t_list[i]))
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
