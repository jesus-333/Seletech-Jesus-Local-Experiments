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

def pair_with_NIRS_sensor_timestamp(NIRS_timestamps_list : list, MiFlora_dataframe, return_difference : bool = False) : 
    """
    Given a list with the timeframes of the NIRS sensor return for each one the nearest MiFlora sample (nearest in time) 

    If return_difference is True return also an array with the difference in seconds between the MiFlora data and associated NIRS measures
    """

    MiFlora_paired_data = pd.DataFrame()
    MiFlora_timestamps_list = list( MiFlora_dataframe['TIMESTAMP'])
    
    difference_list = []

    for i in range(len(NIRS_timestamps_list)):
        NIRS_timestamp = NIRS_timestamps_list[i]

        idx_closest, timestamp_difference = timestamp_functions.get_closest_timestamp(NIRS_timestamp, MiFlora_timestamps_list)

        MiFlora_paired_data = pd.concat([MiFlora_paired_data, MiFlora_dataframe.iloc[idx_closest]], axis = 0)

        difference_list.append(timestamp_difference)

    if return_difference:
        difference_list = pd.DataFrame(difference_list, columns = ['Difference_with_paired_NIRS_timestamp'])
        MiFlora_paired_data = pd.concat([MiFlora_paired_data, difference_list], axis = 1)

    return MiFlora_paired_data


