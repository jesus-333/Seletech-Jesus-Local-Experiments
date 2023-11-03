import numpy as np
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler, RobustScaler

from . import config
from . import manage_data_beans

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def normalize(data, scaler_type : int):
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

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def beans_preprocess_pipeline(dataframe, config : dict, wavelength = None):

    beans_data = manage_data_beans.extract_spectra_from_beans_dataframe(dataframe)

    if config['split_wavelength']: # Do the computation for the 2 mems separately
        if wavelength is None: raise ValueError("If you want to do the computation for the two mems separately you have to pass the wavelength array")
        beans_data_mems_1 = beans_data[:, wavelength <= 1650]
        beans_data_mems_2 = beans_data[:, wavelength >= 1750]

        beans_data_mems_1 = normalize(beans_data_mems_1, config['normalization_type'])
        beans_data_mems_2 = normalize(beans_data_mems_2, config['normalization_type'])

        if config['compute_derivative']:
            beans_data_mems_1 = derivate(beans_data_mems_1, config['derivative_order'])
            beans_data_mems_2 = derivate(beans_data_mems_2, config['derivative_order'])

        beans_data = np.hstack((beans_data_mems_1, beans_data_mems_2))

    else: # Do the computation for the 2 mems together
        beans_data = normalize(beans_data, config['normalization_type'])
        
        if config['compute_derivative']:
            beans_data = derivate(beans_data, config['derivative_order'])
    
    dataframe.iloc[:, 1:len(wavelength) + 1] = beans_data

    return dataframe


