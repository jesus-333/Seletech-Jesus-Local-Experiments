import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from library import manage_data_beans, preprocess

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
path_calib = 'data/Orange/fruit_orange_calib.csv'
path_NON_calib = 'data/Orange/fruit_orange.csv'

compute_absorbance = True
use_sg_preprocess = True

w = 50
p = 3
der = 2

idx = np.random.randint(1920)
idx = 666

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def split_data_per_mems(data_dataframe, remove_mean = False):
    data_mems_1 = data_dataframe.loc[:, "1350":"1650"].to_numpy().squeeze()
    data_mems_2 = data_dataframe.loc[:, "1750":"2150"].to_numpy().squeeze()

    if remove_mean :
        if len(data_mems_1.shape) > 1:
            data_mems_1 = ( data_mems_1.T - data_mems_1.mean(1) ).T
            data_mems_2 = ( data_mems_2.T - data_mems_2.mean(1) ).T
        else :
            data_mems_1 = data_mems_1 - data_mems_1.mean()
            data_mems_2 = data_mems_2 - data_mems_2.mean()

    return data_mems_1, data_mems_2

def compute_preprocess(data, compute_absorbance, use_sg_preprocess):
    data  = preprocess.R_A(data, keep_meta = False)
    data  = preprocess.sg(data, keep_meta = False, w = w, p = p, deriv = der)

    return data

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

data_calib = pd.read_csv(path_calib)
data_non_calib = pd.read_csv(path_NON_calib)

if compute_absorbance or use_sg_preprocess:
    data_calib = compute_preprocess(data_calib, compute_absorbance, use_sg_preprocess)
    data_non_calib = compute_preprocess(data_non_calib, compute_absorbance, use_sg_preprocess)
