"""
Functions used during preprocess of the data

@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler, RobustScaler
from scipy.signal import savgol_filter
import pandas as pd

from . import config
from . import manage_data_beans

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def normalize_sklearn_scaler(data, scaler_type : int):
    """
    Normalize the data with one of the following scaler. The scaler is decided by the scaler type parameter
    StandardScaler (scaler_type = 0) : z = (x - u) / s
    RobustScaler   (scaler_type = 1) : it removes the median and scales the data according to the quantile range (defaults to IQR: Interquartile Range).
                                       The IQR is the range between the 1st quartile (25th quantile) and the 3rd quartile (75th quantile).
    Normalizer     (scaler_type = 2) : Each sample (i.e. each row of the data matrix) with at least one non-zero component is rescaled
                                       independently of other samples so that its norm (l1, l2 or inf) equals one.
    MinMaxScaler   (scaler_type = 3) : Transform features by scaling each feature to a given range.
                                       X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
                                       X_scaled = X_std * (max - min) + min
                                       scaler = MinMaxScaler(feature_range=(-1, 1))

    data must be a numpy array of shape n. spectra x wavelength
    """

    if scaler_type == 0:
        scaler = StandardScaler()
    elif scaler_type == 1:
        scaler = RobustScaler()
    elif scaler_type == 2:
        scaler = Normalizer()
    elif scaler_type == 3:
        scaler = MinMaxScaler()
    else:
        raise ValueError("scaler_type must be an int between 0 and 3 (both included)")

    normalize_data = scaler.fit_transform(data)

    return normalize_data

def derivate(data, n_order : int):
    new_data = np.zeros(data.shape)

    for i in range(data.shape[0]):
        tmp_spectra = data[i]
        for j in range(n_order):
            tmp_spectra = np.gradient(tmp_spectra)

        new_data[i, :] = tmp_spectra

    return new_data

def normalize_with_values_first_column(data, divide_mems : bool = False):
    """
    Normalize dividing each row for the first value of the row
    If divide_mems is True compute the normalization separately for each sensors
    """

    if divide_mems == True:
        data_mems_1 = data.loc[:, "1350":"1650"]
        data_mems_1 = __normalize_with_values_first_column(data_mems_1)
        data.loc[:, "1350":"1650"] = data_mems_1.iloc[:, :]

        data_mems_2 = data.loc[:, "1750":"2150"]
        data_mems_2 = __normalize_with_values_first_column(data_mems_2)
        data.loc[:, "1750":"2150"] = data_mems_2
    else:
        data_both_mems = data.loc[:, "1350":"2150"]
        data_both_mems = __normalize_with_values_first_column(data_both_mems)
        data.loc[:, "1350":"2150"] = data_both_mems

    return data

def __normalize_with_values_first_column(data):
    """
    Normalize the Dataframe by the value of the first column
    """
    data_numpy = data.to_numpy()

    first_value = data_numpy[:, 0]
    data_numpy /= first_value[:, None]

    data.iloc[:, :] = data_numpy

    return data


def normalize_standardization(data, divide_mems : bool = False, remove_mean = True, divide_by_std = True):
    """
    If divide_mems is True compute the normalization separately for each sensors
    """

    if divide_mems == True:
        data_mems_1 = data.loc[:, "1350":"1650"]
        data_mems_1 =  __normalize_standardization(data_mems_1, remove_mean, divide_by_std)
        data.loc[:, "1350":"1650"] = data_mems_1.iloc[:, :]

        data_mems_2 = data.loc[:, "1750":"2150"]
        data_mems_2 =  __normalize_standardization(data_mems_2, remove_mean, divide_by_std)
        data.loc[:, "1750":"2150"] = data_mems_2
    else:
        data_both_mems = data.loc[:, "1350":"2150"]
        data_both_mems =  __normalize_standardization(data_both_mems, remove_mean, divide_by_std)
        data.loc[:, "1350":"2150"] = data_both_mems

    return data

def __normalize_standardization(data, remove_mean, divide_by_std):
    """
    Normalize EACH spectra trough standardization. This mean that each spectra will have mean 0 and std 1.
    N.b. If we take the same wavelength but with different spectra the std between wavelength can still be very high respect the mean between wavelength.
    """
    data_numpy = data.to_numpy()

    if remove_mean:
        mean_data = data_numpy.mean(1)
        data_numpy -= mean_data[:, None]

    if divide_by_std:
        std_data = data_numpy.std(1)
        data_numpy /= std_data[:, None]

    data.iloc[:, :] = data_numpy

    return data


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def sg(data, w = 51, p = 3, deriv = 0, keep_meta = True):
    """
    @author = Dag
    Savitky Golay filter for spectra. Splits spectra into two mems

    window size default 15
    polynomial order 3
    derivative 0
    """
    print("smoothing data")  # 11 #2

    param = {"window_length": w,
             "polyorder": p,
             "deriv": deriv,
             "axis": 1}

    data_id = data.iloc[:, :702].index
    # data_col = data.iloc[:, :702].columns
    mems1 = data.loc[:, "1350":"1650"]
    mems2 = data.loc[:, "1750":"2150"]
    if keep_meta : meta = data.loc[:, "device_id":]
    del data
    sav_mems1 = pd.DataFrame(
        savgol_filter(mems1, **param),
        columns=mems1.columns,
        index=data_id)
    sav_mems2 = pd.DataFrame(
        savgol_filter(mems2, **param),
        columns=mems2.columns,
        index=data_id)
    if keep_meta : data = pd.concat([sav_mems1, sav_mems2, meta], axis=1)
    else : data = pd.concat([sav_mems1, sav_mems2], axis=1)
    print(f'data shape after filter {data.shape}')
    return data

#savisky golay filter,
def R_A(data, keep_meta = True):
    """
    @author = Dag
    Reflectance to absorbance
    """

    data_id = data.iloc[:, :702].index
    # data_col = data.iloc[:, :702].columns
    mems1 = data.loc[:, "1350":"1650"]
    mems2 = data.loc[:, "1750":"2150"]
    if keep_meta : meta = data.loc[:, "device_id":]
    ab_mems1 = np.log10(1/mems1)
    ab_mems2 = np.log10(1/mems2)
    if keep_meta : data = pd.concat([ab_mems1, ab_mems2, meta], axis=1)
    else : data = pd.concat([ab_mems1, ab_mems2], axis=1)

    return data


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Beans experiment function

def beans_preprocess_pipeline(dataframe, config : dict, wavelength = None):

    beans_data = manage_data_beans.extract_spectra_from_beans_dataframe(dataframe)

    if config['split_wavelength']: # Do the computation for the 2 mems separately
        if wavelength is None: raise ValueError("If you want to do the computation for the two mems separately you have to pass the wavelength array")
        beans_data_mems_1 = beans_data[:, wavelength <= 1650]
        beans_data_mems_2 = beans_data[:, wavelength >= 1750]

        beans_data_mems_1 = normalize_sklearn_scaler(beans_data_mems_1, config['normalization_type'])
        beans_data_mems_2 = normalize_sklearn_scaler(beans_data_mems_2, config['normalization_type'])

        if config['compute_derivative']:
            beans_data_mems_1 = derivate(beans_data_mems_1, config['derivative_order'])
            beans_data_mems_2 = derivate(beans_data_mems_2, config['derivative_order'])

        beans_data = np.hstack((beans_data_mems_1, beans_data_mems_2))

    else: # Do the computation for the 2 mems together
        beans_data = normalize_sklearn_scaler(beans_data, config['normalization_type'])

        if config['compute_derivative']:
            beans_data = derivate(beans_data, config['derivative_order'])

    dataframe.iloc[:, 1:len(wavelength) + 1] = beans_data

    return dataframe

def normalize_with_control_group(data, norm_type : int):
    """
    If norm_type == 1 for each spectra remove the mean of the control group, then divide by same mean and multiply by 100
    If norm_type == 2 for each spectra remove the mean of the control group, then divide by the std of the control group
    This operations are made separately for each wavelength.
    N.b. note that this normalization is computed along the wavelength, i.e., each wavelength is normalized respect the values of other wavelengths
    """
    plant_group_list = list(set(data['test_control']))
    data_control = data[data['test_control'] == 'control']
    data_control_mems_1 = data_control.loc[:, "1350":"1650"]
    data_control_mems_2 = data_control.loc[:, "1750":"2150"]

    for plant_group in plant_group_list:
        # Get data for the group
        data_group = data[data['test_control'] == plant_group]

        # Normalize mems1
        data_group_mems_1 = data_group.loc[:, "1350":"1650"]
        data_group_mems_1 = __normalize_with_control_group(data_group_mems_1, data_control_mems_1, norm_type)
        data_group.loc[:, "1350":"1650"] = data_group_mems_1.iloc[:, :]

        # Normalize mems2
        data_group_mems_2 = data_group.loc[:, "1750":"2150"]
        data_group_mems_2 = __normalize_with_control_group(data_group_mems_2, data_control_mems_2, norm_type)
        data_group.loc[:, "1750":"2150"] = data_group_mems_2.iloc[:, :]

        # Save the normalized data
        data[data['test_control'] == plant_group] = data_group

    return data

def __normalize_with_control_group(data_to_normalize, data_control_group, norm_type : int):
    data_numpy_to_normalize = data_to_normalize.to_numpy()
    data_numpy_control_group = data_control_group.to_numpy()

    mean_control = data_numpy_control_group.mean(0)

    if norm_type == 1:
        data_numpy_to_normalize = ( ( data_numpy_to_normalize - mean_control ) / mean_control ) * 100
    elif norm_type == 2:
        std_control = data_numpy_control_group.std(0)
        data_numpy_to_normalize = ( ( data_numpy_to_normalize - mean_control ) / std_control)
    else:
        raise ValueError("norm_type can be only 1 or 2")

    data_to_normalize.iloc[:, :] = data_numpy_to_normalize

    return data_to_normalize

def normalize_with_srs_and_xtalk(data, spectra_srs, spectra_xtalk, percentage_reflectance_srs : float = 0.99):
    normalized_data = data.copy(deep = True)
    spectra_srs_mems_1 = spectra_srs.loc[:, "1350":"1650"].to_numpy()
    spectra_srs_mems_2 = spectra_srs.loc[:, "1750":"2150"].to_numpy()

    spectra_xtalk_mems_1 = spectra_xtalk.loc[:, "1350":"1650"].to_numpy()
    spectra_xtalk_mems_2 = spectra_xtalk.loc[:, "1750":"2150"].to_numpy()

    data_mems_1 = data.loc[:, "1350":"1650"]
    data_mems_1 = __normalize_with_srs_and_xtalk(data_mems_1, spectra_srs_mems_1, spectra_xtalk_mems_1, percentage_reflectance_srs)
    normalized_data.loc[:, "1350":"1650"] = data_mems_1.iloc[:, :]


    data_mems_2 = data.loc[:, "1750":"2150"]
    data_mems_2 = __normalize_with_srs_and_xtalk(data_mems_2, spectra_srs_mems_2, spectra_xtalk_mems_2, percentage_reflectance_srs)
    normalized_data.loc[:, "1750":"2150"] = data_mems_2.iloc[:, :]

    return normalized_data

def __normalize_with_srs_and_xtalk(data_to_normalize, spectra_srs, spectra_xtalk, percentage_reflectance_srs : float) :
    data_numpy_to_normalize = data_to_normalize.to_numpy()

    data_normalized = ( ( data_numpy_to_normalize - spectra_xtalk ) / (spectra_srs - spectra_xtalk) ) * percentage_reflectance_srs

    data_to_normalize.loc[:, :] = data_normalized

    return data_to_normalize
