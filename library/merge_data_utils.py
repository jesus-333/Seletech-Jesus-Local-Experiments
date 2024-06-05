"""
Functions to merge the different data
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from . import preprocess
from . import timestamp_functions

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
    new_dataframe = pd.DataFrame(tmp_data, columns = np.hstack((wavelength.astype(str), 'label', 'label_text')))

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
    If 1 the labels are assigned based on the plants (i.e. ViciaFaba = 0 and PhaseolusVulgaris = 1)

    @param data_beans : (pd.DataFrame) DataFrame with the beans spectra
    @param labels_type : (int) Specify the type of label to create.
    """

    if labels_type == 1 :
        plants_array = data_beans['plant'].to_numpy()
        labels_array = np.zeros(len(plants_array))

        labels_array[plants_array == 'ViciaFaba'] = 0
        labels_array[plants_array == 'PhaseolusVulgaris'] = 1

        numerical_labels_to_text = {0 : 'ViciaFaba', 1 : 'PhaseolusVulgaris'}
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

def create_new_dataframe_orange(data_orange : pd.DataFrame, preprocess_config : dict, rows_to_return : int = -1) :
    """
    Create a new dataframe with the beans data that will be merged with the data from the other sources.
    """
    
    # Get labels for the beans
    labels_array, numerical_labels_to_text = create_label_for_orange(data_orange)
    
    # Convert the data to numpy array and kept only the spectra
    spectra_data, wavelength, idx_spectra = extract_spectra(data_orange, rows_to_return)

    # Get the labels of the selected spectra
    labels_array = labels_array[idx_spectra]
    
    # Create the new dataframe
    new_dataframe = create_new_dataframe_from_single_source(spectra_data, labels_array, numerical_labels_to_text, 'orange', preprocess_config)

    return new_dataframe

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Potos functions

def load_spectra_data_potos(filename, print_var = True, return_numpy : bool = False):
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

    if return_numpy :
        return spectra_plants_numpy, wavelength, timestamp
    else :
        return spectra_plants_df, wavelength, timestamp

def load_water_data_potos(filename, print_var = True):
    log_file_df = pd.read_csv(filename, encoding = 'unicode_escape')
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

def create_extended_water_vector_potos(water_log_timestamp, water_vector, spectra_timestamp):
    """
    Create a vector of index that indicate in the spectra timestamp the closest to when water was given to the plant
    """

    j = 0
    actual_water_timestamp = water_log_timestamp[j]
    water_quantity = water_vector[j]
    extended_water_vector = np.zeros(len(spectra_timestamp))
    w_year, w_month, w_day, w_hour, w_minutes, w_seconds = timestamp_functions.extract_data_from_timestamp(actual_water_timestamp)
    
    for i in range(len(spectra_timestamp)):
        actual_spectra_timestamp = spectra_timestamp[i]
          
        sp_year, sp_month, sp_day, sp_hour, sp_minutes, sp_seconds = timestamp_functions.extract_data_from_timestamp(actual_spectra_timestamp)
          
        if(sp_year == w_year and sp_month == w_month and sp_day == w_day):
            if(sp_hour >= w_hour and sp_minutes >= w_minutes):
                if(water_quantity > 0):
                    extended_water_vector[i] = round(water_quantity/50)
              
                j += 1
                actual_water_timestamp = water_log_timestamp[j]
                water_quantity = water_vector[j]
                w_year, w_month, w_day, w_hour, w_minutes, w_seconds = timestamp_functions.extract_data_from_timestamp(actual_water_timestamp)
    
        if(timestamp_functions.compare_timestamp(actual_spectra_timestamp, actual_water_timestamp)):
            j += 1
            actual_water_timestamp = water_log_timestamp[j]
            water_quantity = water_vector[j]
            w_year, w_month, w_day, w_hour, w_minutes, w_seconds = timestamp_functions.extract_data_from_timestamp(actual_water_timestamp)

    return extended_water_vector

def choose_spectra_based_on_water_V1(extended_water_timestamp, time_interval_start, time_interval_end):
    """
    Create an index vector containg all the spectra after the plant was given water.
    The after is defined with the two variable time_interval_start and time_interval_end and is based on the number of samples.
    N.b. time_interval_end > time_interval_start
    
    # TODO
    If the water was given to the plant at hour XX:YY then the taken spectra will be between XX:YY + time_interval_start and XX:YY  +time_interval_end 
    """
    
    if(time_interval_start >= time_interval_end): raise Exception("time_interval_start should be greater than time_interval_end")
    
    good_idx_tmp = np.zeros(len(extended_water_timestamp))

    good_timestamp = np.where(extended_water_timestamp != 0)[0]

    for idx in good_timestamp : good_idx_tmp[idx + time_interval_start:idx + time_interval_end] = 1
    
    good_idx = good_idx_tmp == 1
    bad_idx = good_idx_tmp != 1
    
    return good_idx, bad_idx

def create_label_for_potos(path_water_file : str, spectra_timestamp, time_interval_start : int, time_interval_end : int) :
    """
    Create the label array for the potos data based on the water given to the plant.
    If the spectra is recored between time_interval_start and time_interval_end after the plant was assigned to wet idx otherwise it is assigned to dry idx.
    """

    water_data, water_timestamp = load_water_data_potos(path_water_file)
    extended_water_timestamp = create_extended_water_vector_potos(water_timestamp, water_data, spectra_timestamp)
    wet_idx, dry_idx = choose_spectra_based_on_water_V1(extended_water_timestamp, time_interval_start = time_interval_start, time_interval_end = time_interval_end)
    
    return wet_idx, dry_idx

def create_new_dataframe_potos(data_potos : pd.DataFrame, spectra_timestamp, path_water_file, time_interval_start : int, time_interval_end : int, preprocess_config : dict, rows_to_return : int = -1) :
    """
    Create a new dataframe with the beans data that will be merged with the data from the other sources.
    """
    
    # Get labels for the beans
    wet_idx, dry_idx  = create_label_for_potos(path_water_file, spectra_timestamp, time_interval_start, time_interval_end)
    
    # Get wet spectra
    spectra_data_wet = data_potos[wet_idx]
    spectra_data_wet, wavelength, _ = extract_spectra(spectra_data_wet, int(rows_to_return / 2))

    # Get dry spectra
    spectra_data_dry = data_potos[dry_idx]
    spectra_data_dry, _, _ = extract_spectra(spectra_data_dry, int(rows_to_return / 2))

    # Concatenate the two arrays and create the labels
    spectra_data = np.vstack((spectra_data_wet, spectra_data_dry))
    labels_array = np.hstack((np.ones(len(spectra_data_wet)), np.zeros(len(spectra_data_dry))))
    spectra_data = pd.DataFrame(spectra_data, columns = wavelength)
    numerical_labels_to_text = {0 : 'dry', 1 : 'wet'}

    # Create the new dataframe
    new_dataframe = create_new_dataframe_from_single_source(spectra_data, labels_array, numerical_labels_to_text, 'potos', preprocess_config)

    return new_dataframe
