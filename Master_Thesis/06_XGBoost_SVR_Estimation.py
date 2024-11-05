#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 20:04:20 2024

@author: ferreira

File will be used to fit ML-Models (XGBoost and SVR)
"""
#%% Section 0: Imports
#%%
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, inspect, text
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from itertools import product
from joblib import Parallel, delayed
import json
import time
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from MLClass import CustomGridSearch, MLfitmodels

# =============================================================================
# Create engines
# =============================================================================
db_path = '00_Data/00_Tables/MLdatasets.db'
db_path_test = '00_Data/00_Tables/MLdatasets_test.db'
db_forecasts_x_path = '00_Data/00_Tables/arma_models_sent.db'
db_path_sent = '00_Data/00_Tables/datasets.db'
engine_sent = create_engine(f'sqlite:///{db_path_sent}')
engine_forecasts_x = create_engine(f'sqlite:///{db_forecasts_x_path}')
engine_test = create_engine(f'sqlite:///{db_path_test}')
engine = create_engine(f'sqlite:///{db_path}')

#%% Section 1. Query and Process Data to fit Model 
#%%
query = """
SELECT *
FROM ml_Lnvar
WHERE "Set Type" = "train"
"""

df_var = pd.read_sql(
    sql = query,
    parse_dates = ['Trading Date'],
    index_col = ['Trading Date'],
    con = engine
    )


# =============================================================================
# NOTA BENE: WE DO NOT CONSIDER THE MODELS WITH COVARIATE ANYMORE
# SO ANY INFORMATION TO SENTIMENT INDEX CAN BE DESREGARDED.

# WITH COVARIATE WE MEAN EXTERNAL COVARIATES. OUR ANALYSIS IS UNIVARIATE.
# =============================================================================
# =============================================================================
# We will also import the estimation for the covariate (sent_index) made
# with our ARMA model. These estimations will be used as the observation in
# the validation set. The sentiment index enters the function contemporanously,
# meaning that, whenever we forcast y_t+i, for all i, we also need a forecast
# of the covariate.

# We start by import the sentiment index itself (the observed values computed
# with finBert and pca. And join them with the df_var data set.) As long
# as we do not pass any covariate to the function call (in fitting the models), 
# they will just be ignored.
# =============================================================================

query = """
SELECT "Trading Date", "pca_sent_index"
FROM "SentIndex"
"""
df_sentiment = pd.read_sql(
    sql = query,
    parse_dates = ['Trading Date'],
    index_col = ['Trading Date'],
    con = engine_sent
    )

df_var = pd.merge(df_var.reset_index(), 
                  df_sentiment.reset_index(), 
                  on = 'Trading Date',
                  how = 'left')

# =============================================================================
# Set the trading date to be the index column again
# =============================================================================
df_var.set_index('Trading Date', inplace = True)

query = """
SELECT "Origin", "h.1", "h.2", "h.3", "h.4", "h.5"
FROM "forecast_and_target"
WHERE "type" = "forecast" AND "params" = "(2, 0, 2)"
"""

df_forecasted_sent = pd.read_sql(
    sql = query,
    con = engine_forecasts_x,
    index_col = ['Origin'],
    parse_dates = ['Origin']
    )

# =============================================================================
# We will rename the index column of the df_forecasted_sent to Trading Date.
# Nota bene: in this case origin means that day the forecast was made, and
# the columns h.1, h.2, h.3, ... are the 1, 2, ..., step ahead forecast made
# at that day.
# =============================================================================


df_forecasted_sent.rename_axis('Trading Date', inplace = True)

#%% Section 2: Model Grid Search Specifcation
#%%

param_grid_svr = {
    'C': [1, 10, 25, 50],
    'epsilon': [0.005, 0.01, 0.5, 0.8],
    'kernel': ['linear', 'rbf'],
}


param_grid_xgboost = {
    'booster': ['gbtree'],
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'min_child_weight': [3, 7],
    'random_state': [19921123]
}

#%% 2.3. Fit the models: Only Lagged Target Values as Covariate.
#%% 2.3.1. Support Vector Regression
#%%

# =============================================================================
# result = MLfitmodels(df_var = df_var, 
#               securities = ['DAX'], 
#               vol_proxies = ['Squared Return',
#                              'Adj. Realized Volatility',
#                              'Realized Volatility'],
#               param_grid = param_grid_svr,
#               db_engine =  engine,
#               skip_fitted = True,
#               model_class = 'SVR',
#               lags = [1],
#               h = 5,
#               min_size = 250,
#               x_covariates = None,
#               use_standardscaler = True)
# =============================================================================

#%% 2.3.2. XGBoost
#%%
# =============================================================================
# result = MLfitmodels(df_var = df_var, 
#               securities = ['DAX', 'SP500'], 
#               vol_proxies = ['Squared Return', 
#                               'Adj. Realized Volatility', 
#                               'Realized Volatility'],
#               param_grid = param_grid_xgboost,
#               db_engine =  engine,
#               skip_fitted = True,
#               model_class = 'XGBRegressor',
#               lags = [1],#, 2, 3, 4, 5],
#               h = 5,
#               min_size = 250,
#               x_covariates = None,
#               use_standardscaler = True)
# =============================================================================



#%% 2.4. Fit the Models: Sentiment Index as Extra Covariate
#%% 2.4.1. SVR
#%%

# =============================================================================
# result = MLfitmodels(df_var = df_var, 
#               securities = ['DAX', 'SP500'], 
#               vol_proxies = ['Squared Return', 
#                               'Adj. Realized Volatility', 
#                               'Realized Volatility'],
#               param_grid = param_grid_svr,
#               db_engine =  engine,
#               skip_fitted = True,
#               model_class = 'SVR',
#               lags = [1, 2, 3, 4, 5],
#               h = 5,
#               min_size = 250,
#               x_covariates = ['pca_sent_index'],
#               df_forecasts_x = df_forecasted_sent,
#               use_standardscaler = True)
# =============================================================================

#%% 2.4.2. XGBoost
#%%


# =============================================================================
# result = MLfitmodels(df_var = df_var, 
#               securities = ['DAX', 'SP500'], 
#               vol_proxies = ['Squared Return', 
#                               'Adj. Realized Volatility', 
#                               'Realized Volatility'],
#               param_grid = param_grid_xgboost,
#               db_engine =  engine,
#               skip_fitted = True,
#               model_class = 'XGBRegressor',
#               lags = [1, 2, 3, 4, 5],
#               h = 5,
#               min_size = 250,
#               x_covariates = ['pca_sent_index'],
#               df_forecasts_x = df_forecasted_sent,
#               use_standardscaler = True)
# 
# =============================================================================


