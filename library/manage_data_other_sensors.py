"""
Read and manage the data collected through non-NIRS sensors

@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import pandas as pd
import numpy as np

from . import timestamp_functions

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# MiFlora sensor

def read_data_MiFlora(path : str, return_numpy : bool = False) :
    data = pd.read_csv(path)

    old_timestamps = data['TIMESTAMP']
    data['TIMESTAMP'] = timestamp_functions.convert_timestamps_format_1(list(old_timestamps)) 

    return data

def pair_with_NIRS_sensor_timestamp(NIRS_timestamp, MiFlora_datframe, return_difference = False) : 
    """
    Given a list with the timeframes of the NIRS sensor return for each one the nearest MiFlora sample (nearest in time) 
    """
    pass
