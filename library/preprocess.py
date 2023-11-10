import numpy as np
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler, RobustScaler
from scipy.signal import savgol_filter
import pandas as pd

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

def sg(data, w =51, p =3, deriv=0):
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
    meta = data.loc[:, "device_id":]
    del data
    sav_mems1 = pd.DataFrame(
        savgol_filter(mems1, **param),
        columns=mems1.columns,
        index=data_id)
    sav_mems2 = pd.DataFrame(
        savgol_filter(mems2, **param),
        columns=mems2.columns,
        index=data_id)
    data = pd.concat([sav_mems1, sav_mems2, meta], axis=1)
    print(f'data shape after filter {data.shape}')
    return data

#savisky golay filter,
def R_A(data):
    """
    @author = Dag
    Reflectance to absorbance
    """

    data_id = data.iloc[:, :702].index
    # data_col = data.iloc[:, :702].columns
    mems1 = data.loc[:, "1350":"1650"]
    mems2 = data.loc[:, "1750":"2150"]
    meta = data.loc[:, "device_id":]
    ab_mems1 = np.log10(1/mems1)
    ab_mems2 = np.log10(1/mems2)
    data = pd.concat([ab_mems1, ab_mems2, meta], axis=1)
    
    return data

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def compute_mdwi(data, return_as_dataframe = True):
    selected_band = data.loc[:, "1350":"1750"].to_numpy()
    R_max = np.max(selected_band, 1)
    R_min = np.min(selected_band, 1)

    mdwi = (R_max - R_min)/(R_max + R_min)

    if return_as_dataframe:
        return pd.DataFrame(mdwi, columns = ["mdwi"])
    else:
        return mdwi


def compute_ndni_1(data, band, normalized_index = True, return_as_dataframe = True):
    if band == 'lower': 
        R_ref = data.loc[:, "1650"].to_numpy() # Theoretically it should be 1680... take 1650 due to data limitation
        R_abs = data.loc[:, "1450"].to_numpy()
    elif band == 'upper':   
        R_ref = data.loc[:, "2100"].to_numpy()
        R_abs = data.loc[:, "1940"].to_numpy()
    else:                   
        raise ValueError("The parameter band must be lower or upper")

    if normalized_index:
        ndni = (R_ref - R_abs) / (R_ref + R_abs)
    else:
        ndni = R_abs / R_ref

    if return_as_dataframe:
        return pd.DataFrame(ndni, columns = ["ndni"])
    else:
        return ndni


def compute_ndni_2(data, is_absorbance = True, return_as_dataframe = True):
    """
    Compute the ndni according to the second presented formula. 
    Note that due to data limitation I have to take the 1650 wavelength
    """

    if is_absorbance:
        A_1510 = data.loc[:, "1510"].to_numpy()
        A_1680 = data.loc[:, "1650"].to_numpy()
    else:
        R_1510 = data.loc[:, "1510"].to_numpy()
        R_1680 = data.loc[:, "1650"].to_numpy()

        A_1510 = np.log10(1/R_1510)
        A_1680 = np.log10(1/R_1680)

    ndni = (A_1510 - A_1680) / (A_1510 + A_1680)

    if return_as_dataframe:
        return pd.DataFrame(ndni, columns = ["ndni"])
    else:
        return ndni

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


