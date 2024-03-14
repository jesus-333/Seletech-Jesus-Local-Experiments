"""
Compute the t-test between a range of wavelength
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import pandas as pd
from scipy.stats import ttest_ind, ttest_rel
import numpy as np

from library import manage_data_beans, preprocess

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# plant = 'PhaseolusVulgaris'
plant = 'ViciaFaba'

lamp_power = None

# Parameter to remove spectra with amplitude to low
min_amplitude = 1000
percentage_above_threshold = 0.8

# Range of wavelength to use in the t-test
wavelength_min = 1520
wavelength_max = 1540

# Parameter for preprocess
compute_absorbance = True
use_SNV = True
use_sg_filter = True
w = 50
p = 3
deriv = 2

mems_to_plot = 1

plot_config = dict(
    figsize = (20, 12),
    fontsize = 20,
    split_plot = False,
    save_fig = True,
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def get_data(t, plant, lamp_power, compute_absorbance, use_SNV, use_sg_filter, w, p, deriv):
    # Load data
    path_spectra = "data/beans/t{}/csv/beans_avg.csv".format(t)
    spectra_data_ALL, wavelength, group_labels_list, plant_labels_list, plant_type_list = manage_data_beans.read_data_beans_single_file(path_spectra, return_numpy = False)
    spectra_data = spectra_data_ALL[spectra_data_ALL['gain_0'] == 1]
    if plant is not None : spectra_data = spectra_data[spectra_data['plant'] == plant]
    if lamp_power is not None : spectra_data = spectra_data[spectra_data['lamp_0'] == lamp_power]

    # Remove spectra with at least a portion of spectra below the threshold
    spectra_data, idx_data_to_keep = preprocess.filter_spectra_by_threshold(spectra_data, threshold = min_amplitude, percentage_above_threshold = percentage_above_threshold)

    # Preprocess
    meta_data = spectra_data.loc[:, "timestamp":"type"]
    if compute_absorbance : spectra_data = preprocess.R_A(spectra_data, keep_meta = False)
    if use_SNV : spectra_data = preprocess.normalize_standardization(spectra_data, divide_mems = True)
    if use_sg_filter : spectra_data = preprocess.sg(spectra_data, w, p, deriv, keep_meta = False)
    if compute_absorbance or use_sg_filter: # Since during preprocess the metadata are removed here are restored
        spectra_data = pd.concat([meta_data, spectra_data], axis = 1)

    return spectra_data

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

t_list = [0, 1, 2, 3, 4, 5]
group_list = ['control', 'test_150', 'test_300']
p_value_array = np.zeros(len(group_list) * len(t_list), len(group_list) * len(t_list))
combo_string_list = []

idx_1 = 0
for i_1 in range(len(t_list)):
    t_1 = t_list[i_1]

    # Load data (1)
    spectra_data_1 = get_data(t_1, plant, lamp_power, compute_absorbance, use_SNV, use_sg_filter, w, p, deriv)

    for j_1 in range(len(group_list)):
        group_1 = spectra_data_1[spectra_data_1['test_control'] == group_list[j_1]]
        group_1 = group_1.loc[:, str(wavelength_min):str(wavelength_max)].to_numpy()

        group_1 = group_1.mean(0)

        idx_2 = 0
        for i_2 in range(len(t_list)):
            t_2 = t_list[i_2]

            # Load data (2)
            spectra_data_2 = get_data(t_2, plant, lamp_power, compute_absorbance, use_SNV, use_sg_filter, w, p, deriv)

            for j_2 in range(len(group_list)):
                group_2 = spectra_data_2[spectra_data_2['test_control'] == group_list[j_2]]
                group_2 = group_2.loc[:, str(wavelength_min):str(wavelength_max)]
                group_2 = group_2.mean(0)
                
                # Compute t-test
                t_test_output = ttest_ind(group_1, group_2, equal_var = False)
                t_statistics, p_value = t_test_output.statistic, t_test_output.pvalue
                
                # Save results
                p_value_array[idx_1, idx_2] = p_value
                idx_2 += 1

        idx_1 += 1
        combo_string_list.append("t{} - {}".format(t_1, group_list[j_1]))
