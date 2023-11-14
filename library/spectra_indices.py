"""
Functions to compute different indices based on specific wavelength

@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)
"""

import numpy as np

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
