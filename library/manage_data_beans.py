import pandas as pd
import numpy as np

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def read_data_beans_single_file(path : str, return_numpy : bool = False):
    data = pd.read_csv(path)

    wavelength = data.keys()[1:-10].to_numpy(dtype = float)

    if return_numpy:
        # Extract only the spectra and return them
        return extract_spectra_from_beans_dataframe(data), wavelength
    else:
        # Extract spectra, labels of control subjcet, labels of plants with salt and plants' namea
        group_labels_list = list(set(data['test_control'])) # Specify the macro group (control, test_150, test_300)
        plant_labels_list = list(set(data['type']))     # Since each group has multiple plants this is the list with the label/name for each plant (CON1, CON2, ... , NACL150_1, ... etc)
        plant_type_list = list(set(data['plant']))     # Get the scientific name of the 2 plants used in the experiment

        return data, wavelength, group_labels_list, plant_labels_list, plant_type_list

def extract_spectra_from_beans_dataframe(dataframe):
    spectra_data = dataframe.iloc[:, 1:-10].to_numpy()

    return spectra_data

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Average computation per group/subgroup

def compute_average_and_std_per_group(dataframe, group_labels_list : list):
    """
    Compute the average for the 3 main group: control, test_150, test_300
    """
    mean_array = np.zeros((len(group_labels_list), 702))
    std_array  = np.zeros((len(group_labels_list), 702))
    for i in range(len(group_labels_list)):
        group_label = group_labels_list[i]

        group_dataframe = dataframe[dataframe['test_control'] == group_label]

        mean_array[i] = extract_spectra_from_beans_dataframe(group_dataframe).mean(0)
        std_array[i] = extract_spectra_from_beans_dataframe(group_dataframe).std(0)

    return mean_array, std_array

def compute_average_and_std_per_subgroup(dataframe, subgroup_labels_list : list):
    """
    Compute the average for the each subgroup: CON1, CON2, etc
    """
    mean_dict = {}
    std_dict  = {}
    for i in range(len(subgroup_labels_list)):
        subgroup_label = subgroup_labels_list[i]

        subgroup_dataframe = dataframe[dataframe['type'] == subgroup_label]

        mean_dict[subgroup_label] = extract_spectra_from_beans_dataframe(subgroup_dataframe).mean(0)
        std_dict[subgroup_label] = extract_spectra_from_beans_dataframe(subgroup_dataframe).std(0)

    return mean_dict, std_dict


