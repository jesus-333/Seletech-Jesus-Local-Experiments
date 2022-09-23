#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 14:43:48 2022

@author: Alberto Zancanaro (jesus)
@organization: University of Padua
"""

#%%

import sys
sys.path.insert(0, 'support')

from support.training_wandb_V2 import create_and_log_model

#%%

model_config = dict(
    neurons_per_layer = [64, 128, 36],
    hidden_space_dimension = 2,
    use_as_autoencoder = True,
    use_cnn = False,
    print_var = True
)