#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 21:33:37 2024

@author: LenonFerreira-Exacta
"""

#%% Section 0: Imports
#%%
# =============================================================================
# Set tensorflow-environment options to make sure that results are consistent
# (i.e., sometimes results could be different due to rounding, the option
# ensures that this does not happen.)
# =============================================================================
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, inspect
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
import tensorflow as tf
import itertools
import json
from tqdm import tqdm
import time
from datetime import datetime
from ClassLSTM import LSTMModel

devices = tf.config.experimental.list_physical_devices('GPU')

if len(devices) > 0:
    tf.config.experimental.set_visible_devices([], 'GPU')

#%% Section 1: Dataset imports
#%%

# =============================================================================
# Iterate over all the databases in the directory and obtain the table names
# =============================================================================
tables_dir = '00_Data/00_Tables/'
databases = [
    db for db in os.listdir(tables_dir) if os.path.splitext(db)[1] == '.db'
    ]

# =============================================================================
# create connections to the database and store them in a dictionary
# =============================================================================
con_dict = {
    key: create_engine('sqlite:///{}'.format(tables_dir + key)) for key in databases
}

tables_dict = {
    os.path.basename(key): inspect(engine).get_table_names()
    for key, engine in con_dict.items()
}

# =============================================================================
# create engine to store the results of the training process
# =============================================================================

engine_deep = create_engine(f'sqlite:///{tables_dir + "LSTM.db"}')

#%%

# =============================================================================
# Import the dataset with the log-variance
# =============================================================================

query = """
SELECT "Trading Date", "Realized Volatility", "Adj. Realized Volatility", "Squared Return", "Security"
FROM ml_Lnvar
WHERE "Set Type" = "train"
"""


df_var = pd.read_sql(
    query, 
    con = con_dict['MLdatasets.db'],
    parse_dates = ['Trading Date']
)




# =============================================================================
# set the date column to be the index of the dataframe
# =============================================================================
df_var.set_index('Trading Date', inplace = True)

#%% Section 2: Model Training
#%% Section 2.1.: Model Training without Covariate
#%%


lstm = LSTMModel(df = df_var, random_seed = 19921123)

# =============================================================================
# Uncomment the lines below to re-estimate the models.
# =============================================================================
df_history = lstm.estimate_model(target_variables = ['Squared Return',
                                                      'Adj. Realized Volatility'],
                    securities = ['SP500', 'DAX'],
                    h = 5,
                    lags = [1, 2, 3, 4, 5],
                    dropout = [0, 0.2],
                    layers = [32, 64],
                    epochs = [1, 10, 20],
                    batches = [1, 32, 64],
                    val_size = [0, ‚0.2],
                    to_db = engine_deep,
                    table_name = 'lstm_training_results')




