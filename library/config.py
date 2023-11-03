"""
Get various dictionary with the complete config for the different steps of the analysis
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
import numpy as np
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def get_config_preprocess():
    """
    Get the config for the preprocess pipeline
    """

    config = dict(
        split_wavelength = True,
        normalization_type = 1,
        compute_derivative = True,
        derivative_order = 1,
    )

    return config


def get_wavelengths_mems():
    """
    Get the two vectors with the wavelengths of the two mems
    """

    wavelengths = dict(
        mems_1 = np.arange(1350, 1650 + 1),
        mems_2 = np.arange(1750, 2150 + 1)
    )

    return wavelengths
