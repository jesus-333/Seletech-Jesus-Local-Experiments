"""
Functions to merge the different data
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from library import preprocess

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# General functions

def extract_spectra(data : pd.DataFrame, rows_to_return : int = -1) :
    """
    Extract the spectra from the dataframes, removing the metadata columns.
    If rows_to_return is specified the function will return that number of rows.
    """

    wavelength = np.hstack((np.arange(1350, 1650 + 1), np.arange(1750, 2150 + 1)))

    data = data.loc[:, "1350":"2150"]
    
    idx = np.arange(data.shape[0])
    if rows_to_return > 0 :
        idx = np.random.choice(idx, rows_to_return, replace = False)
        data = data.iloc[idx]

    return data, wavelength, idx

def create_new_dataframe_from_single_source(spectra_data, labels : np.ndarray, numerical_labels_to_text : dict, source_origin : str, preprocess_config : dict):
    """
    Given the spectra and the labels from a specific source creaste a Pandas DataFrame that will be merged with the dataset from the other sources.
    A column with the source origin (e.g. beans) will be added to the dataframe.
    A column with th label in textual form will be added to the dataframe.
    """
    if preprocess_config['compute_absorbance'] : spectra_data = preprocess.R_A(spectra_data, keep_meta = False)
    if preprocess_config['use_SNV'] :
        # spectra_data = preprocess.normalize_standardization(spectra_data, divide_mems = True)
        spectra_data = preprocess.normalize_sklearn_scaler(spectra_data, scaler_type = 0)
    if preprocess_config['use_sg_filter'] : spectra_data = preprocess.sg(spectra_data, preprocess_config['w'], preprocess_config['p'], preprocess_config['deriv'], keep_meta = False)
    if preprocess_config['use_minmax'] : spectra_data = preprocess.normalize_sklearn_scaler(spectra_data, scaler_type = 3)

    # Remove border artifacts caysed by the SG filter
    if preprocess_config['use_sg_filter'] :
        spectra_data, wavelength = remove_border_artifacts(spectra_data, preprocess_config['w'])
    else : 
        wavelength = np.hstack((np.arange(1350, 1650 + 1), np.arange(1750, 2150 + 1)))
    
    labels_text = np.array([numerical_labels_to_text[label] for label in labels])
    tmp_data = np.hstack((spectra_data, labels.reshape(-1, 1), labels_text.reshape(-1, 1)))
    new_dataframe = pd.DataFrame(tmp_data, columns = np.hstack((wavelength, 'label', 'label_text')))

    return new_dataframe

def remove_border_artifacts(spectra_data, w) :
    # Compute the waves where the boreder artifacts appear
    tmp_wave_1_mems_1 = int(1350 + w / 2)
    tmp_wave_2_mems_1 = int(1650 - w / 2)
    tmp_wave_1_mems_2 = int(1750 + w / 2)
    tmp_wave_2_mems_2 = int(2150 - w / 2)

    # Kept only the spectra inside the range
    tmp_spectra_mems_1 = spectra_data.loc[:, str(tmp_wave_1_mems_1):str(tmp_wave_2_mems_1)]
    tmp_spectra_mems_2 = spectra_data.loc[:, str(tmp_wave_1_mems_2):str(tmp_wave_2_mems_2)]
    spectra_data = pd.concat([tmp_spectra_mems_1, tmp_spectra_mems_2], axis = 1)
    
    # Update the wavelength array
    wavelength = np.hstack((np.arange(tmp_wave_1_mems_1, tmp_wave_2_mems_1 + 1), np.arange(tmp_wave_1_mems_2, tmp_wave_2_mems_2 + 1)))
    
    return spectra_data, wavelength

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Beans functions

def merge_beans_spectra(t_list = []) :
    """
    Take the data of the beans from different days and merge them together
    """

    merge_dataset = pd.DataFrame()

    for i in range(len(t_list)) :
        t = t_list[i]
        path_spectra = "data/beans/t{}/csv/beans.csv".format(t)

        tmp_dataset = pd.read_csv(path_spectra)

        merge_dataset = pd.concat([merge_dataset, tmp_dataset], axis = 0)

    return merge_dataset


def create_label_for_beans(data_beans : pd.DataFrame, labels_type : int) :
    """
    Create the label for the beans type based on labels_type.
    If 1 the labels are assigned based on the plants (e.g. ViciaFaba = 1 and PhaseolusVulgaris = 2)

    @param data_beans : (pd.DataFrame) DataFrame with the beans spectra
    @param labels_type : (int) Specify the type of label to create.
    """

    if labels_type == 1 :
        plants_array = data_beans['plant'].to_numpy()
        labels_array = np.zeros(len(plants_array))

        labels_array[plants_array == 'ViciaFaba'] = 1
        labels_array[plants_array == 'PhaseolusVulgaris'] = 2

        numerical_labels_to_text = {1 : 'ViciaFaba', 2 : 'PhaseolusVulgaris'}
    else :
        raise ValueError('labels_type must have value 1. Current value is {}'.format(labels_type))

    return labels_array, numerical_labels_to_text

def create_new_dataframe_beans(data_beans : pd.DataFrame, labels_type : int, preprocess_config : dict, rows_to_return : int = -1) :
    """
    Create a new dataframe with the beans data that will be merged with the data from the other sources.
    """
    
    # Get labels for the beans
    labels_array, numerical_labels_to_text = create_label_for_beans(data_beans, labels_type)
    
    # Convert the data to numpy array and kept only the spectra
    spectra_data, wavelength, idx_spectra = extract_spectra(data_beans, rows_to_return)

    # Get the labels of the selected spectra
    labels_array = labels_array[idx_spectra]
    
    # Create the new dataframe
    new_dataframe = create_new_dataframe_from_single_source(spectra_data, labels_array, numerical_labels_to_text, 'beans', preprocess_config)

    return new_dataframe

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Orange functions

def create_label_for_orange(data_orange : pd.DataFrame) :
    """
    Create the label for the beans type based on labels_type.

    @param data_orange : (pd.DataFrame) DataFrame with the orange spectra
    """

    # possible_labels_orange = ['orange', 'orangeDOWN_whiteUP', 'white', 'whole_orange']
    orange_section_array = data_orange['part'].to_numpy()
    labels_array = np.zeros(len(orange_section_array))

    labels_array[orange_section_array == 'orange'] = 1
    labels_array[orange_section_array == 'orangeDOWN_whiteUP'] = 2
    labels_array[orange_section_array == 'white'] = 3
    labels_array[orange_section_array == 'whole_orange'] = 4

    numerical_labels_to_text = {1 : 'orange', 2 : 'orangeDOWN_whiteUP', 3 : 'white', 4 : 'whole_orange'}

    return labels_array, numerical_labels_to_text

def create_new_dataframe_orange(data_orange : pd.DataFrame, labels_type : int, preprocess_config : dict, rows_to_return : int = -1) :
    """
    Create a new dataframe with the beans data that will be merged with the data from the other sources.
    """
    
    # Get labels for the beans
    labels_array, numerical_labels_to_text = create_label_for_beans(data_orange, labels_type)
    
    # Convert the data to numpy array and kept only the spectra
    spectra_data, wavelength, idx_spectra = extract_spectra(data_orange, rows_to_return)

    # Get the labels of the selected spectra
    labels_array = labels_array[idx_spectra]
    
    # Create the new dataframe
    new_dataframe = create_new_dataframe_from_single_source(spectra_data, labels_array, numerical_labels_to_text, 'beans', preprocess_config)

    return new_dataframe

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Potos functions

def load_spectra_data(filename, print_var = True):
    spectra_plants_df = pd.read_csv(filename, header = 2, encoding= 'unicode_escape')
    
    # Convert spectra dataframe in a numpy matrix
    spectra_plants_numpy = spectra_plants_df.iloc[:, 5:-1].to_numpy(dtype = float)
    if(print_var): print("Spectra matrix shape\t=\t", spectra_plants_numpy.shape, "\t (Time x Wavelength)")
    
    # Recover wavelength
    wavelength = spectra_plants_df.keys()[5:-1].to_numpy(dtype = float)
    if(print_var): print("Wavelength vector shape\t=\t", wavelength.shape)
    
    # Recover timestamp
    timestamp = spectra_plants_df['# Timestamp'].to_numpy()
    if(print_var): print("Timestamp vector shape\t=\t", timestamp.shape)
    
    return spectra_plants_numpy, wavelength, timestamp

#%% Water related information

def load_water_data(filename, print_var = True):
    if print_var: print("Water File Loaded\n")
    
    # Extract timestamp (in the same format of the spectra file)
    tmp_data = log_file_df['Date']
    tmp_hour = log_file_df['Time']
    log_timestamp = []
    for data, hour in zip(tmp_data, tmp_hour):
      tmp_timestamp = data.split('/')[2] + '_' + data.split('/')[1] + '_' + data.split('/')[0] + '_'
      tmp_timestamp += hour.split(':')[0] + '_' + hour.split(':')[1] + '_00'
    
      log_timestamp.append(tmp_timestamp)
    
    # Extract daily water
    water = np.asarray(log_file_df['H2O[g]'])
    water[np.isnan(water)] = 0
    
    log_timestamp = np.asarray(log_timestamp)[2:]
    water = water[2:]
    
    if print_var: print("Water vector length =\t" , len(water))
    if print_var: print("Log timestamp length =\t" , len(log_timestamp), "\n")
    
    return water, log_timestamp
