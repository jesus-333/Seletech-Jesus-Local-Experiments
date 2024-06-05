"""
Count the data from each source before extracting a subsample from each source for the merging

"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import pandas as pd

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings

path_beans = "data/merged_dataset/beans.csv"
path_orange = "data/merged_dataset/orange.csv"
path_potos = "data/merged_dataset/potos.csv"

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Read data

data_beans  = pd.read_csv(path_beans, index_col = 0)
data_orange = pd.read_csv(path_orange, index_col = 0)
data_potos  = pd.read_csv(path_potos, index_col = 0)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Create the string to print with the information from each source

string_to_print = "Total number of samples for each source:\n"
string_to_print += "\tBeans: {}\n".format(data_beans.shape[0])
string_to_print += "\tOrange: {}\n".format(data_orange.shape[0])
string_to_print += "\tPotos: {}\n".format(data_potos.shape[0])

string_to_print += "\n- - - - - - - - - - - - - - - -\n"
string_to_print += "Division of sammple for Beans:\n"
for label in set(data_beans["label_text"]) :
    string_to_print += "\t{}: {}\n".format(label, data_beans[data_beans["label_text"] == label].shape[0])

string_to_print += "\n- - - - - - - - - - - - - - - -\n"
string_to_print += "Division of sammple for Orange:\n"
for label in set(data_orange["label_text"]) :
    string_to_print += "\t{}: {}\n".format(label, data_orange[data_orange["label_text"] == label].shape[0])

string_to_print += "\n- - - - - - - - - - - - - - - -\n"
string_to_print += "Division of sammple for Potos:\n"
for label in set(data_potos["label_text"]) :
    string_to_print += "\t{}: {}\n".format(label, data_potos[data_potos["label_text"] == label].shape[0])

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

print(string_to_print)
