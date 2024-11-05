#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 21:25:56 2024

@author: ferreira

Foreword: This file will be used exclusively to estimate the (G)Arch-family of
Models. We created a class named GarchModels and stored in the file:
    GarchModelClass.py.
    
This class allows us to fit a large number of models and cross-validate them.
"""


#%% Section 0: Imports
#%%
import pandas as pd
import os
import numpy as np
from sqlalchemy import create_engine
import sqlite3 as sqlite
import tqdm as tqdm
from GarchModelsClass import GarchModels
import warnings

if __name__ == '__main__':
# =============================================================================
# Create sql-engines to store each one of the results to a local database.
# =============================================================================
    tables_dir = '00_Data/00_Tables/'
    engines_dict = {
        'sp500': {
            'arch': 'sqlite:///{}'.format(tables_dir + 'ArchRes_sp500.db'),
            'garch': 'sqlite:///{}'.format(tables_dir + 'GarchRes_sp500.db'),
            'egarch': 'sqlite:///{}'.format(tables_dir + 'EgarchRes_sp500.db'),
            'figarch': 'sqlite:///{}'.format(tables_dir + 'FigarchRes_sp500.db')
            },
        'dax': {
            'arch': 'sqlite:///{}'.format(tables_dir + 'ArchRes_dax.db'),
            'garch': 'sqlite:///{}'.format(tables_dir + 'GarchRes_dax.db'),
            'egarch': 'sqlite:///{}'.format(tables_dir + 'EgarchRes_dax.db'),
            'figarch': 'sqlite:///{}'.format(tables_dir + 'FigarchRes_dax.db')
            }
        }



# =============================================================================
# Import the data used in the estimation
# =============================================================================

    engine_data = 'sqlite:///{}'.format(tables_dir + 'datasets.db')
    
    query = """
    SELECT *
    FROM "Agg. Dataset"
    WHERE "Set Type" = "train"
    """
    
    df_agg = pd.read_sql(
        sql = query,
        con = engine_data,
        parse_dates = 'Trading Date',
        index_col = ['Trading Date']
    )
    
    
    df_sp500 = df_agg[df_agg['Security'] == 'SP500']
    
    df_dax = df_agg[df_agg['Security'] == 'DAX']


# =============================================================================
# Obtain returns and volatilities
# =============================================================================

    df_return_sp500 = df_sp500['Daily Return']
    df_return_dax = df_dax['Daily Return']
    
    vol_cols = ['Squared Return', 'Realized Volatility', 'Adj. Realized Volatility']
    columns_to_keep = [
        col for col in df_agg.columns if col in vol_cols
        ]
    
    df_vol_sp500 = df_sp500[columns_to_keep]
    
    df_vol_dax = df_dax[columns_to_keep]


#%%
#%% Section 4: Estimation
#%% Section 4.1. Arch-Model Estimation 
#%% Section 4.1.1. S&P500 Case
#%%

# =============================================================================
# Instantiate the class and start the computation. The results of the estima-
# tion will automatically be saved at the very end.
# =============================================================================

    # gm = GarchModels(df = df_return_sp500, max_comb = 31)
    
    # gm.Cross_validation_evaluation(
    #     df_var = df_vol_sp500,
    #     cv_type = 'origin',
    #     h = 5,
    #     size = 250,
    #     seed = 19921123,
    #     max_workers = 10,
    #     loss_funcs = ['mse', 'mae', 'mape'],
    #     mean = ['Constant', 'Zero'],
    #     vol = ['ARCH'],
    #     pqo = (30, 0, 0),
    #     pqo_ismax = True,
    #     dist = ['skewt', 't', 'normal', 'ged'],
    #     to_db = True,
    #     engine = engines_dict['sp500']['arch']
    #     )

#%% Section 4.1.2. DAX Case
#%%

    # gm = GarchModels(df = df_return_dax, max_comb = 31)
    
    # gm.Cross_validation_evaluation(
    #     df_var = df_vol_dax,
    #     cv_type = 'origin',
    #     h = 5,
    #     size = 250,
    #     seed = 19921123,
    #     max_workers = 10,
    #     loss_funcs = ['mse', 'mae', 'mape'],
    #     mean = ['Constant', 'Zero'],
    #     vol = ['ARCH'],
    #     pqo = (30, 0, 0),
    #     pqo_ismax = True,
    #     dist = ['skewt', 't', 'normal', 'ged'],
    #     to_db = True,
    #     engine = engines_dict['dax']['arch']
    #     )
    
#%% Section 4.2. Garch-Model Estimation
#%% Section 4.2.1. The S&P500 Case
#%%
    # print("=" * 50)
    # print('Estimation of GARCH for S&P500: Begin')
    # gm = GarchModels(df = df_return_sp500, max_comb = 31)
    
    # gm.Cross_validation_evaluation(
    #     df_var = df_vol_sp500,
    #     cv_type = 'origin',
    #     h = 5,
    #     size = 250,
    #     seed = 19921123,
    #     max_workers = 10,
    #     loss_funcs = ['mse', 'mae', 'mape'],
    #     mean = ['Constant', 'Zero'],
    #     vol = ['GARCH'],
    #     pqo = (5, 5, 5),
    #     pqo_ismax = True,
    #     dist = ['skewt', 't', 'normal', 'ged'],
    #     to_db = True,
    #     engine = engines_dict['sp500']['garch']
    #     )
    # print('Estimation of GARCH for S&P500: End')
    # print("=" * 50)
    

#%% Section 4.2.2. The DAX Case
#%%
    # print("=" * 50)
    # print('Estimation of GARCH for DAX: Begin')
    # gm = GarchModels(df = df_return_dax, max_comb = 31)
    
    # gm.Cross_validation_evaluation(
    #     df_var = df_vol_dax,
    #     cv_type = 'origin',
    #     h = 5,
    #     size = 250,
    #     seed = 19921123,
    #     max_workers = 10,
    #     loss_funcs = ['mse', 'mae', 'mape'],
    #     mean = ['Constant', 'Zero'],
    #     vol = ['GARCH'],
    #     pqo = (5, 5, 5),
    #     pqo_ismax = True,
    #     dist = ['skewt', 't', 'normal', 'ged'],
    #     to_db = True,
    #     engine = engines_dict['dax']['garch']
    #     )
    
    # print('Estimation of GARCH for DAX: End')
    # print("=" * 50)
    
#%% Section 4.3. EGARCH-Model Estimation
#%% Section 4.3.1. S&P500 Case
#%%

    # print("=" * 50)
    # print('Estimation of EGARCH for S&P500: Begin')
    
    # gm = GarchModels(df = df_return_sp500, max_comb = 31)
    
    # gm.Cross_validation_evaluation(
    #     df_var = df_vol_sp500,
    #     cv_type = 'origin',
    #     h = 5,
    #     size = 250,
    #     seed = 19921123,
    #     max_workers = 10,
    #     loss_funcs = ['mse', 'mae', 'mape'],
    #     mean = ['Constant', 'Zero'],
    #     vol = ['EGARCH'],
    #     pqo = (5, 5, 5),
    #     pqo_ismax = True,
    #     dist = ['skewt', 't', 'normal', 'ged'],
    #     to_db = True,
    #     engine = engines_dict['sp500']['egarch']
    #     )

    # print('Estimation of EGARCH for S&P500: End')
    # print("=" * 50)

#%% Section 4.3.2. DAX Case
#%%

    # print("=" * 50)
    # print('Estimation of EGARCH for DAX: Begin')
    # gm = GarchModels(df = df_return_dax, max_comb = 31)
    
    # gm.Cross_validation_evaluation(
    #     df_var = df_vol_dax,
    #     cv_type = 'origin',
    #     h = 5,
    #     size = 250,
    #     seed = 19921123,
    #     max_workers = 10,
    #     loss_funcs = ['mse', 'mae', 'mape'],
    #     mean = ['Constant', 'Zero'],
    #     vol = ['EGARCH'],
    #     pqo = (5, 5, 5),
    #     pqo_ismax = True,
    #     dist = ['skewt', 't', 'normal', 'ged'],
    #     to_db = True,
    #     engine = engines_dict['dax']['egarch']
    #     )
    
    # print('Estimation of EGARCH for DAX: End')
    # print("=" * 50)
#%% Section 4.4. FIGARCH Estimation
#%% Section 4.4.1. The S&P500 Case
#%%
# =============================================================================
#     print("=" * 50)
#     print('Estimation of FIGARCH for S&P500: Begin')
# 
#     gm = GarchModels(df = df_return_sp500, max_comb = 31)
#     
#     gm.Cross_validation_evaluation(
#         df_var = df_vol_sp500,
#         cv_type = 'origin',
#         h = 5,
#         size = 250,
#         seed = 19921123,
#         max_workers = 10,
#         loss_funcs = ['mse', 'mae', 'mape'],
#         mean = ['Constant', 'Zero'],
#         vol = ['FIGARCH'],
#         pqo = (1, 1, 0),
#         pqo_ismax = True,
#         dist = ['skewt', 't', 'normal', 'ged'],
#         to_db = True,
#         engine = engines_dict['sp500']['figarch']
#         )
#     
#     print('Estimation of FIGARCH for S&P500: End')
#     print("=" * 50)
# 
# =============================================================================
#%% Section 4.4.2. The DAX Case
#%%
# =============================================================================
#     print("=" * 50)
#     print('Estimation of FIGARCH for DAX: Begin')
#     gm = GarchModels(df = df_return_dax, max_comb = 31)
#     
#     gm.Cross_validation_evaluation(
#         df_var = df_vol_dax,
#         cv_type = 'origin',
#         h = 5,
#         size = 250,
#         seed = 19921123,
#         max_workers = 10,
#         loss_funcs = ['mse', 'mae', 'mape'],
#         mean = ['Constant', 'Zero'],
#         vol = ['FIGARCH'],
#         pqo = (1, 1, 0),
#         pqo_ismax = True,
#         dist = ['skewt', 't', 'normal', 'ged'],
#         to_db = True,
#         engine = engines_dict['dax']['figarch']
#         )
#     
#     print('Estimation of FIGARCH for DAX: End')
#     print("=" * 50)
# =============================================================================
