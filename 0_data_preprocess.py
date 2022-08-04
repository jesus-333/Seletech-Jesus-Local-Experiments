#%% Imports 

import sys
sys.path.insert(0, 'support')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
import time

from support.datasets import load_spectra_data, load_water_data, create_extended_water_vector 
from support.timestamp_function import convert_timestamps_in_dataframe
from support.preprocess import aggregate_HT_data_V1, aggregate_HT_data_V2

#%% Load data
normalize_trials = -1

spectra_plants_numpy, wavelength, spectra_timestamp = load_spectra_data("data/[2021-08-05_to_11-26]All_PlantSpectra.csv", normalize_trials)

water_data, water_timestamp = load_water_data("data/[2021-08-05_to_11-26]PlantTest_Notes.csv")
extended_water_timestamp = create_extended_water_vector(water_timestamp, water_data, spectra_timestamp)

ht_data = pd.read_csv("data/[2021-08-05_to_11-26]All_PlantHTSensor.csv", encoding= 'unicode_escape')
humidity = ht_data[' Humidity[%]']
temperature = ht_data[' Temperature[C]']
ht_timestamp = ht_data['Timestamp']

#%% Convert timestamp in dataframe

spectra_timestamp_dataframe = convert_timestamps_in_dataframe(spectra_timestamp)
spectra_timestamp_dataframe.to_csv("data/jesus_spectra_timestamp.csv") 

ht_timestamp_dataframe = convert_timestamps_in_dataframe(ht_timestamp)
ht_timestamp_dataframe.to_csv("data/jesus_ht_timestamp.csv") 


#%%
spectra_timestamp_numpy = spectra_timestamp_dataframe.to_numpy()[:, 0:5]
ht_timestamp_numpy = ht_timestamp_dataframe.to_numpy()[:, 0:5]

spectra_plants_dataframe = pd.DataFrame(spectra_plants_numpy, columns = wavelength)
pd.concat([spectra_timestamp_dataframe, spectra_plants_dataframe], axis = 1).to_csv("data/jesus_spectra_full_dataset.csv") 

ht_dataframe_full = pd.concat([ht_timestamp_dataframe, humidity, temperature], axis = 1)
ht_dataframe_full.to_csv("data/jesus_ht_full_dataset.csv") 

#%%

spectra_timestamp_numpy = spectra_timestamp_dataframe.to_numpy()
ht_timestamp_numpy = ht_timestamp_dataframe.to_numpy()

#%% Test aggregation of ht data

start_time = time.time()
aggregate_h_array, aggregate_t_array, aggregate_timestamp = aggregate_HT_data_V2(ht_timestamp_numpy, spectra_timestamp_numpy, humidity.to_numpy(), temperature.to_numpy())
tot_time =  time.time() - start_time

print("Total time aggregation (V2):", round(tot_time, 2))


# plt.plot(humidity)
# plt.plot(temperature)
plt.figure(figsize = (15, 10))
plt.plot(aggregate_h_array)
plt.plot(aggregate_t_array)
# plt.xlim([90000, 130000])

#%% Test performance Pandas vs SQL

# start_time = time.time()

# for i in range(spectra_timestamp_numpy.shape[0]):
#     year, month, day, hour, minute, second = spectra_timestamp_numpy[i]
#     a = ht_dataframe_full['year'] == year
#     b = ht_dataframe_full['month'] == month
#     c = ht_dataframe_full['day'] == day  
#     d = ht_dataframe_full['hour'] == hour 
#     e = ht_dataframe_full['minutes'] == minute
    
#     a = ht_dataframe_full.loc[a & b & c & d & e]
    
#     if i/spectra_timestamp_numpy.shape[0] > 1/100: break
    
# tot_time =  time.time() - start_time
# print("Performance Pandas:")
# print("\tEnd time (1%): ", round(tot_time, 2))
# print("\tEnd time (estimate total): ", round(tot_time * 100, 2), "\n")


# con = sqlite3.connect('data/SQL/test_db.db')
# cur = con.cursor()
# # cur.execute('SELECT * FROM jesus_ht_full_dataset where day = 5')
# # data = cur.fetchall()

# start_time = time.time()

# for i in range(spectra_timestamp_numpy.shape[0]):
#     year, month, day, hour, minute, second = spectra_timestamp_numpy[i]
#     sql_string = "SELECT * FROM jesus_spectra_full_dataset WHERE year = {} AND month = {} AND day = {} AND hour = {} AND minutes = {}".format(year, month, day, hour, minute)
#     cur.execute(sql_string)
#     data = cur.fetchall()
    
#     if i/spectra_timestamp_numpy.shape[0] > 1/100: break
    
# tot_time =  time.time() - start_time
# print("Performance SQL:")
# print("\tEnd time (3%): ", tot_time)
# print("\tEnd time (estimate total): ", tot_time * 100)