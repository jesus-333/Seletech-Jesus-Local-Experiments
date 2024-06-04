"""
Functions to merge the different data
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import pandas as pd

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

t_list = [0, 1, 2]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

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

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

merge_dataset_beans = merge_beans_spectra(t_list = t_list)


