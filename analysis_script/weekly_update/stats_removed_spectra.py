import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from library import manage_data_beans, preprocess

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

threshold = 1000

t_list = [0, 1, 2, 3, 4, 5]

gain = 1

plant = 'PhaseolusVulgaris'
# plant = 'ViciaFaba'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def filter_spectra_by_threshold(spectra_dataframe, threshold : int):
    spectra_data = spectra_dataframe.loc[:, "1350":"2150"].to_numpy().squeeze()

    # idx_data_to_remove = ((spectra_data < threshold).sum(1) >= 75)
    idx_data_to_keep = ((spectra_data > threshold).sum(1) >= 600) # Tieni lo spettro solo se 650 delle 702 lunghezze d'onda registrate sono superiori alla threshold
    
    return spectra_dataframe[idx_data_to_keep], idx_data_to_keep

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

info_to_print = "Plant {}\n".format(plant)
specific_plant_removed_count = dict()

for i in range(len(t_list)):
    t = t_list[i]

    # Load data
    path_spectra = "data/beans/t{}/csv/beans_avg.csv".format(t)
    spectra_data_ALL, wavelength, group_labels_list, plant_labels_list, plant_type_list = manage_data_beans.read_data_beans_single_file(path_spectra, return_numpy = False)
    spectra_data_ALL = spectra_data_ALL[spectra_data_ALL['gain_0'] == gain]
    if plant is not None : spectra_data_ALL = spectra_data_ALL[spectra_data_ALL['plant'] == plant]
    
    # Get the list with all the different plants
    if i == 0:
        tmp_plant_type_list = list(set(spectra_data_ALL['type']))
        for tmp_plant_type in tmp_plant_type_list : specific_plant_removed_count[tmp_plant_type] = 0
        del tmp_plant_type, tmp_plant_type_list

    # Remove spectra with lower value
    spectra_kept, idx_data_to_keep = filter_spectra_by_threshold(spectra_data_ALL, threshold)
    
    # Get the removed spectra
    idx_data_to_remove = np.logical_not(idx_data_to_keep)
    spectra_removed = spectra_data_ALL[idx_data_to_remove]
    
    # Percentage of spectra kept and removed
    percentage_kept = len(spectra_kept) / len(spectra_data_ALL) * 100
    percentage_removed = len(spectra_removed) / len(spectra_data_ALL) * 100

    info_to_print += "Day t = {}\n".format(t)
    info_to_print += "\tPercentage spectra kept    = {:.2f}%\n".format(percentage_kept)
    info_to_print += "\tPercentage spectra removed = {:.2f}%\n".format(percentage_removed)


    # Get lamp power removed spectra
    lamp_power_removed = list(set(spectra_removed['lamp_0']))
    
    info_to_print += "\tDistribution of lamp power in removed spectra:\n"
    for lamp_power in lamp_power_removed : 
        tmp_spectra = spectra_removed[spectra_removed['lamp_0'] == lamp_power]
        tmp_percentage = len(tmp_spectra) / len(spectra_removed) * 100
        info_to_print += "\t\tLamp power {} : {:.2f}% ({}/{}) of the removed spectra\n".format(lamp_power, tmp_percentage, len(tmp_spectra), len(spectra_removed))
        
        # Count the plants removed
        # Power lamp 50 is always removed. So I count only with lamp power different from 50
        if lamp_power != 50 : 
            tmp_plant_type_list = list(set(tmp_spectra['type']))
            for tmp_plant_type in tmp_plant_type_list : specific_plant_removed_count[tmp_plant_type] += 1

    del tmp_spectra, tmp_percentage, tmp_plant_type_list, tmp_plant_type

    info_to_print += "\n"

info_to_print += "Times plant removed : \n"
for specific_plant in specific_plant_removed_count:   
    info_to_print += "\t{} : {}\n".format(specific_plant, specific_plant_removed_count[specific_plant])

info_to_print += "\n"

print(info_to_print)

