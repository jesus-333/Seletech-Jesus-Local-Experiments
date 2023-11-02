import matplotlib.pyplot as plt
import os

from library import manage_data_beans

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

t_list = [0, 1, 2, 3, 4, 5, 6] # Different day (see path)

# plant = 'PhaseolusVulgaris'
plant = 'ViciaFaba'

plot_config = dict(
    figsize = (25, 40),
    fontsize = 15,
    save_fig = True
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

group_to_idx_dict = {'control' : 0, 'test_150' : 1, 'test_300' : 2}

plt.rcParams.update({'font.size': plot_config['fontsize']})
fig, axs = plt.subplots(len(t_list), 3, figsize = plot_config['figsize'])

for i in range(len(t_list)):
    path = "data/beans/t{}/csv/beans.csv".format(t_list[i])

    data, wavelength, group_labels_list, plant_labels_list, plant_type_list = manage_data_beans.read_data_beans_single_file(path, return_numpy = False)

    data = data[data['plant'] == plant]

    mean_array, std_array = manage_data_beans.compute_average_and_std_per_group(data, group_labels_list)

    for j in range(len(group_labels_list)):

        idx = group_to_idx_dict[group_labels_list[j]]
        print(group_labels_list[j], idx)

        ax = axs[i, idx]
        ax.plot(wavelength, mean_array[idx], label = group_labels_list[idx])
        ax.fill_between(wavelength, mean_array[idx] + std_array[idx], mean_array[idx] - std_array[idx], alpha = 0.25)
        ax.set_ylim([750, 2750])
        ax.set_xlim([1350, 2150])
        ax.set_title('{} - t{}'.format(group_labels_list[idx], t_list[i]))

fig.tight_layout()
fig.show()


if plot_config['save_fig']:
    path_save = 'Saved Results/beans_spectra/'
    os.makedirs(path_save, exist_ok = True)

    path_save = 'Saved Results/beans_spectra/'
    path_save += 'average_spectra_different_group'
    fig.savefig(path_save + ".png", format = 'png')
    fig.savefig(path_save + ".pdf", format = 'pdf')
