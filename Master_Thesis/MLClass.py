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

#%% Section 1: Class Defintion
#%%

class CustomGridSearch():
    


    def __init__(self,
                 model_class,
                 param_grid,
                 min_size,
                 h,
                 keep_forecast = False):

# =============================================================================
# Drop NA-values from the dataset
# =============================================================================
        self.model_class = model_class
        self.param_grid = param_grid
        self.min_size = min_size
        self.h = h
        self.scaler_x = None
        self.scaler_y = None
        self.df_forecasts = None
        self.setup_logging()
        self.keep_forecast = keep_forecast

    def setup_logging(self):
# =============================================================================
# Function created for debugging purpose.
# =============================================================================
        logging.basicConfig(
            filename = 'debug.log',
            level = logging.DEBUG,
            format = '%(asctime)s - %(levelname)s - %(message)s',
            filemode = 'w'
            )
    def CV_RollingFromOrigin(self, df:pd.DataFrame, min_size:int, h:int):
        """
        A generator to create train and validation split
        for cross validation by using the rolling-forecasting-origin method        
        
        Nota Bene: Make sure that the data set is in ascending order regarding
        the dates.
        
        Parameters
        ----------
        df : cudf.DataFrame
            A cudf dataframe with the data to be used in the cross-validation.
        min_size : int
            The minimum-size of datapoints needed to estimate the model.
        h : int
            The horizon of the prediction (i.e. the size of the validation set)
    
        Returns
        -------
        A generator for train and validation splits. The data is returned as a
        one dimensional numpy array.
    
        """
        
    
    
        assert df.shape[1] == 1, " ".join([
            'Data-Frame must contain a single column'
        ])
    
        data = df.to_numpy().ravel()
      
# =============================================================================
# iterate over the dataset and make sure to have enough data to create the va-
# lidation set, which is always h-ahead of the train set.
# =============================================================================
        for period in range(len(data) - min_size - h + 1):
# =============================================================================
# The data is added sequentially and the validation split moves with period.
# =============================================================================
            train_split = data[ :min_size + period]
            val_split = data[min_size + period: min_size + period + h]
            
            yield train_split, val_split



    def fit_model(self, 
                  df, 
                  y_col, 
                  x_col = None,
                  df_forecasts_x = None,
                  lags = 5, 
                  use_standardscaler = True,
                  window_size = 5,
                  log_to_level = True):
      
        
# =============================================================================
# Make sure that if covariates where passed, that a matrix with forecasted
# values to be used as the train observation for the covariate is passed.

# The covariate matrix must have the predictions h.1, ..., h.x ahead it its
# columns. And the columns must be named accordingly.

# The index column should contain the date. The covariate itself must be passed
# in the same data frame as the target!
# =============================================================================
        
        if x_col is not None:
            assert isinstance(df_forecasts_x, pd.DataFrame), " ".join([
                'The forecasted_cov arg must be a pandas dataframe containing',
                'the forecasts for the covariate.'
                ])
            
            assert pd.api.types.is_datetime64_ns_dtype(df_forecasts_x.index), " ".join([
                'Index column of the forecasted_cov dataframe must be in datatime64[ns] format'
            ])   
            
            self.x_col = x_col[0]
            self.x_covariate = x_col[0]
            self.df_forecasts_x = df_forecasts_x
            
        else:
            self.df_forecasts_x = None
            self.x_col = None
            
            
        best_score_overall = float('inf')
        best_score_one_step = float('inf')
        best_params_overall = None
        best_params_one_step = None
        df = df.copy()
 
 
        self.y_col = y_col[0]
        
        assert pd.api.types.is_datetime64_ns_dtype(df.index), " ".join([
            'Index column must be in datatime64[ns] format'
        ])   
# =============================================================================
# If use_standardscaler = True, the code will search for non-dummy variables
# in the data. Any column containing fewer than 3 different values will be
# treated as containing a dummy and will not be standardized.
# Nota Bene: 3 allows for the existence of Na.
# =============================================================================
        
        continuous_columns = [
            column for column in df[x_col].columns.tolist() 
            if df[column].nunique() > 3
            and column != y_col
        ] if x_col is not None else []
        
        df.sort_index(axis = 0, ascending = True, inplace = True)



# =============================================================================
# Create lagged values of the taget variable
# =============================================================================
        lag_columns = []
        
        for i in range(1, lags + 1):
            
            lag_columns.extend(['L{}{}'.format(i, self.y_col)])
            df['L{}{}'.format(i, self.y_col)] = df[self.y_col].shift(i)
    
    
# =============================================================================
# Make sure that the lagged-values are the first in the data set and that they
# are part of the 'x_col'
# =============================================================================
        self.x_col = lag_columns + x_col if x_col else lag_columns

# =============================================================================
# Drop all NA-Values from the data set.
# =============================================================================
        df.dropna(inplace = True)
   
# =============================================================================
# Generate all combinations of the parameters
# =============================================================================
        param_combinations = self.generate_grid_combination(self.param_grid)
        
        def evaluate_params(params):
            
            self.setup_logging()
           
            model = self.model_class(**params)
                        
            scores = {
                'one-step': [],
                'overall': []
            }
 
            forecasts_dict = {'forecast': [], 'target': []}
    
            counter = 0         
            cv_gen = list(
                self.CV_RollingFromOrigin(df = df.index.to_frame(), 
                                          min_size = self.min_size, 
                                          h = self.h)
                )
            
            df_forecasts = self.df_forecasts
           
            for train_indices, test_indices in cv_gen:
# =============================================================================
# Create train-test data frames and save them to dictionaries according to the
# arguments passed in the yx dict
# =============================================================================
    
                X_train, X_test = (
                    df[self.x_col].loc[train_indices], 
                    df[self.x_col].loc[test_indices] 
                )
                    
              
  

  
                y_train, y_test = (
                    df[self.y_col].loc[train_indices], 
                    df[self.y_col].loc[test_indices]
                )

# =============================================================================
# Scale the variables (if applicable)
# =============================================================================
    
                if use_standardscaler:
                    self.scaler_x = StandardScaler()
                    self.scaler_y = StandardScaler()
                    
                    x_cols_transform = (
                        lag_columns + continuous_columns
                        ) if continuous_columns else lag_columns
                      
                    y_train = self.scaler_y.fit_transform(
                        y_train.values.reshape(-1, 1)
                    ).flatten()
                      
                    y_test = self.scaler_y.transform(
                        y_test.values.reshape(-1, 1)
                    ).flatten()
                    
                    X_train_continuous = self.scaler_x.fit_transform(
                        X_train.loc[:, x_cols_transform]
                    )
                      
                    X_test_continuous = self.scaler_x.transform(
                        X_test.loc[:, x_cols_transform]
                    )
                    

# =============================================================================
# Replace the contiuous columns with the scaled values
# =============================================================================
                    
                    X_train.loc[:, x_cols_transform] = X_train_continuous
                    X_test.loc[:, x_cols_transform] = X_test_continuous

# =============================================================================
# Notice that the prediction generated by the model is a one-step ahead predi-
# ction. As we would like to obtain up to five steps ahead predictions, we will
# need to use an interactive method (by replacing the last prediction as the
# lagged values in the next periods)
# =============================================================================

          
# =============================================================================
# Use the data to compute forecasts
# ============================================================================= 

                model.fit(X_train.values, y_train)
                forecasts = self.multi_step_ahead_prediction(
                    model = model,
                    X_test = X_test,
                    X_train = X_train,
                    h = self.h,
                    lags = lags
                )
                
            
# =============================================================================
# Convert the y_test data set to its original scale (if applicable)
# If dependent variable is in log and user passed the argument log_to_level 
# "True", the dependent variable we apply the exponential function to the
# variable.
# =============================================================================

                y_test = (
                    self.scaler_y.inverse_transform(y_test.reshape(-1, 1))
                    ) if use_standardscaler else y_test
            
    
                if counter == 0:
                    
                    df_forecasts = pd.DataFrame.from_dict(forecasts)
                    df_forecasts.set_index('Trading Date', inplace = True)
                    
                    df_forecasts = df_forecasts.astype(
                        {col: np.float64 for col in df_forecasts.columns 
                         if col != 'Trading Date'}
                        )
# =============================================================================
# create a data-set to store the target variable
# =============================================================================
                    df_target = df_forecasts.copy()
                    
                    
                    df_target.loc[
                        df_forecasts.index[0], [f'h.{i}' for i in range(1, self.h + 1)]
                    ] = (
                        y_test.T.astype(np.float64) if use_standardscaler 
                        else y_test.values.T.astype(np.float64) 
                        )
# =============================================================================
# Apply the inverse transformation to the columns of the df_forecast
# (if applicable)
# =============================================================================
 
                    df_forecasts = (
                        pd.DataFrame(
                            self.scaler_y.inverse_transform(df_forecasts),
                            columns = df_target.columns,
                            index = df_target.index
                        ) if use_standardscaler else df_forecasts
                    )
                    
                    
                    counter += 1
                else:
            
                    df_forecasts_temp = pd.DataFrame.from_dict(forecasts)
                    df_forecasts_temp.set_index('Trading Date', inplace = True)
                    df_forecasts_temp = df_forecasts_temp.astype(
                        {col: np.float64 for col in df_forecasts_temp.columns 
                         if col != 'Trading Date'}
                        )                   
# =============================================================================
# create a data-set to store the target variable
# =============================================================================
                    df_target_temp = df_forecasts_temp.copy()
                    
                   
                    df_target_temp.loc[
                        df_forecasts_temp.index[0], 
                        [f'h.{i}' for i in range(1, self.h + 1)]
                    ] = (
                        y_test.T.astype(np.float64) if use_standardscaler 
                        else y_test.values.T.astype(np.float64)
                        ) 
# =============================================================================
# Apply the inverse transformation to the columns of the df_forecast
# (if applicable)
# =============================================================================

                    df_forecasts_temp = (
                        pd.DataFrame(
                            self.scaler_y.inverse_transform(df_forecasts_temp),
                            columns = df_target_temp.columns,
                            index = df_target_temp.index
                        ) if use_standardscaler else df_forecasts_temp
                    )
                
                
                    df_forecasts = pd.concat([df_forecasts, df_forecasts_temp], axis = 0)
                    df_target = pd.concat([df_target, df_target_temp], axis = 0)

                    
      
            df_forecasts.sort_index(ascending = True, inplace = True)
            df_target.sort_index(ascending = True, inplace = True)


# =============================================================================
# Transform the variables to their level-values if applicable (i.e. log > normal)
# =============================================================================  
          
            df_forecasts = (
                np.exp(df_forecasts)
                ) if log_to_level else df_forecasts
            
            df_target = np.exp(df_target) if log_to_level else df_target
 
            
            if self.keep_forecast:
                
                forecasts_dict['forecast'].append(df_forecasts)
                forecasts_dict['target'].append(df_target)
                self.forecasts = forecasts_dict
                
           
# =============================================================================
# Compute mean squared error for the 1-step ahead and for the overall prediction
# =============================================================================     
            mse_one_step = mean_squared_error(
                df_target['h.1'], df_forecasts['h.1']
            )
              
            mse_overall = mean_squared_error(
                df_target, df_forecasts
            )
              
            scores['one-step'].append(mse_one_step)
            scores['overall'].append(mse_overall)
                       

            return params, scores, forecasts_dict if self.keep_forecast else None
        
        
        results = Parallel(n_jobs = -1)(
            delayed(evaluate_params)(params) for params in tqdm(
                param_combinations,
                desc = 'Cross-Validating Models', 
                unit = 'it', 
                ncols = 100
                )
            )
        
        
        forecasts_list = [result[2] for result in results if result[2] is not None]
        transformed_data = []
        for result in results:
            params_dict, scores_dict, _ = result
            
# =============================================================================
# Convert the parameter dict to a string value.
# =============================================================================
            params_str = json.dumps(params_dict)  # Convert params dictionary to string
            
            for i in range(len(scores_dict['one-step'])):
                
                transformed_data.append({
                    'params': params_str,
                    'one-step': scores_dict['one-step'][i],
                    'overall': scores_dict['overall'][i]
                })
# =============================================================================
# create a pandas dataframe with the specification in one column
# and the mse in another column, and return the dataframe
# =============================================================================
        df_result = pd.DataFrame(transformed_data)
# =============================================================================
# Obtain optimal models return a dictionary with the specifications,
# the scores and the optimal models
# =============================================================================
        min_overall_index = df_result['overall'].idxmin()
        best_overall_spec = df_result.loc[min_overall_index].values
        

        min_one_step_index = df_result['one-step'].idxmin()
        best_one_step_spec = df_result.loc[min_one_step_index].values
        
        final_dict = {
            'specification': df_result,
            'opt_params': (best_one_step_spec[0], best_overall_spec[0]),
            'mse': (best_one_step_spec[1], best_overall_spec[1]),
            'forecasts_list': forecasts_list
            }
        

        return final_dict


    def filter_svr_params(self, combination):
        kernel = combination.get('kernel')
    
        if kernel == 'linear':
            

            combination.pop('degree', None)
            combination.pop('gamma', None)
            combination.pop('coef0', None)
    
        if kernel == 'poly':
            combination.pop('gamma', None)
            combination.pop('coef0', None)
    
        if kernel in ['rbf', 'sigmoid']:
            combination.pop('degree', None) 
            combination.pop('coef0', None) 
      
        
        return combination


    def generate_grid_combination(self, param_grid):
        
        keys, values = zip(*param_grid.items())
        combinations = set()
        for value_combination in product(*values):
 
            combination = dict(zip(keys, value_combination))
      
            if self.model_class == SVR:
                
                combination = self.filter_svr_params(combination)

# =============================================================================
# Ensure that combinations are unique
# =============================================================================
            combinations.add(frozenset(combination.items()))

        return [dict(combination) for combination in combinations]

    def multi_step_ahead_prediction(self, model, X_test, X_train, h, lags):
        #logging.debug('Starting Forecast')
        dict_results = {'Trading Date': X_train.index[-1]}

            
        X_test = X_test.copy()
        X_train = X_train.copy()
        predictions = []

# =============================================================================
# The first data point for the 1-step ahead prediction is the last data point
# in the train data set. (Here considering only the lagged y-values.)
# =============================================================================
        current_data = X_train.iloc[-1].copy()

# =============================================================================
# Obtain lagged values in the test_set, these are the first lags columns. 
# =============================================================================
        df_lags = X_test.iloc[:, :lags]
        
        df_exog_covariates = X_test.iloc[:, lags:]
        
       
        exog_is_empty = df_exog_covariates.empty
# =============================================================================
# Use the last data point in the train-set to obtain the 1-step ahead prediction.
# Notice that if a covariate was passed, we will need to replace its value
# by its forecasted value.
# =============================================================================
        if not exog_is_empty:

            x_forecasted = self.df_forecasts_x.loc[X_train.index[-1], 'h.1']
       
            current_data = current_data.to_frame()
            
            current_data = current_data.T

            x_column_index = current_data.columns.get_loc(self.x_covariate)
            current_data.iloc[0, x_column_index] = x_forecasted
                   
            
        input_features = current_data.values.reshape(1, -1)
        
        prediction = model.predict(input_features)
        predictions.append(prediction)
        
        dict_results['h.1'] = prediction
        
        steps = h
        if steps > 1:
# =============================================================================
# For the 2-step ahead prediction, the last seen data is the first observation
# in the test-set (replacing the first lagged-observation by the last 
# prediction).
# =============================================================================     

            input_features_lags = df_lags.iloc[0].values
            
            input_features_lags[0] = prediction.item()
        
# =============================================================================
# the forecast for the exogeneous variable will be the h2-step ahead forecasts
# which we made with the ARMA(2, 2) model
# =============================================================================
            if not exog_is_empty:
               
                x_forecasted = self.df_forecasts_x.loc[X_train.index[-1], 'h.2']

                input_features_exog = np.array([x_forecasted])

                
                input_features = np.concatenate((input_features_lags, input_features_exog))
                #logging.debug('input_featues_created')
                
            else:
                input_features = input_features_lags

        
# =============================================================================
# Iterate over the steps and do repeat more or less the process above
# =============================================================================
     
        for i in range(1, h):
            # i = 4, h = 5
# =============================================================================
# Make new prediction with the newly defined input_features
# =============================================================================   
            prediction = model.predict(input_features.reshape(1, -1))

            dict_results[f'h.{i + 1}'] = prediction
# =============================================================================
# Roll the input_features_lags by 1 index and add new prediction as the first
# value.
# ============================================================================= 
            input_features_lags = np.roll(input_features_lags, 1)
        
            input_features_lags[0] = prediction.item()

# =============================================================================
# Obtain new set of exogeneous variables. Notice that we assumed that the future
# values of the exogeneous variables from the df_forecasts_x dataset.
# =============================================================================

        if not exog_is_empty:
            
            if i < h-1:
            
               
                x_forecasted = self.df_forecasts_x.loc[X_train.index[-1], f'h.{i + 2}']
                input_features_exog = x_forecasted
                
# =============================================================================
# Create new set of input futures to predict the next step
# =============================================================================
            input_features = np.concatenate(
                (input_features_lags, input_features_exog)
            )
        else:
            input_features = input_features_lags
            
       


# =============================================================================
# return dictionary with forecasts
# =============================================================================
        #logging.debug('Prediction successful.')
        return dict_results

#%% 2. Class Implementation
#%% 2.1. Function to fit the models:
#%%

def MLfitmodels(df_var, 
              securities,
              vol_proxies,
              param_grid,
              db_engine,
              skip_fitted = True,
              model_class = 'SVR',
              lags = [1, 2, 3, 4, 5],
              h = 5,
              window_size = 5,
              min_size = 250,
              x_covariates = None,
              df_forecasts_x = None,
              use_standardscaler = False,
              keep_forecast = False):

    
    df_exception = pd.DataFrame()
# =============================================================================
# Assert that security, and vol_proxies and covariates exist inside of the
# dataset and that they are passed as a list
# =============================================================================
    db_engine = db_engine
    df_var = df_var.copy()
    args = [vol_proxies, securities, lags]
    
    
    
    table_name = '{}_GridSearchValuation'.format(model_class.upper())
    table_name = table_name if x_covariates is None else table_name + 'WithCovariate'
    
    
    
    for arg in args:
        assert isinstance(arg, list), " ".join([
            'securities, vol_proxies, and lags must be passed as lists.'
            ])
        
    for security in securities:
        assert (
            security in df_var['Security'].tolist()
                ), " ".join([
            'Security: {}'.format(security),
            'not available in the column "Security" of the data set.'
            ])
           
        
    for vol_proxy in vol_proxies:
        assert (
            vol_proxy in df_var.columns.tolist()
            ), " ".join([
                'Vol Proxy {}'.format(vol_proxy),
                'not present in the columns of the data set.'
                ])
        
    if x_covariates is not None:
        
        for x_covariate in x_covariates:
            
            assert (
                x_covariate in df_var.columns.tolist()
                ), " ".join([
                    "covariate: {}".format(x_covariate),
                    "not in the columns of the data set."
                    ])
                    
                    
    df_fitted = pd.DataFrame()
# =============================================================================
# If user wants to skip the combination of lags, vol proxy and security that is
# already stored in the data base:
# =============================================================================
    if skip_fitted and db_engine is not None:
        try:
            inspector = inspect(db_engine)
            
            if table_name in inspector.get_table_names():
                
                query = """
                SELECT *
                FROM "{}"
                """.format(table_name)
                
                
                df_fitted = pd.read_sql(
                    sql = query,
                    con = db_engine
                    )
                
                
# =============================================================================
# Create a column containing the Volatility Proxy, the Lags, and the Security
# =============================================================================
                df_fitted['vol_lags_security'] = (
                    df_fitted['Vol Proxy'] + '_' + df_fitted['lags'].astype(str)
                    + '_' + df_fitted['Security']
                    ).str.lower()
                
                
            
            else:
      
                skip_fitted = False
        except:
            
            skip_fitted = False
        
# =============================================================================
# If db-engine is none, we need to ensure that only one parameter is passed
# in the lags, security, vol proxy and in the params-grid   
# =============================================================================
    if db_engine is None:
        assert (
            len(lags) == len(securities) == len(vol_proxies) == 1
            ), " ".join([
                'If a database is not passed, only a combination of security',
                'lag and volatility proxy is allowed.'
                ])
# =============================================================================
# Iterate over the values of the param_grid and make sure that the keys have
# only one element as values
# =============================================================================
        for value in list(param_grid.values()):
            
            assert len(value) == 1, " ".join([
                'If a database is not passed, the lists in the param_grid',
                'cannot contain more than 1 element.'
                ])
            
        skip_fitted = False
        
    model_dict = {
        'SVR': SVR,
        'XGBRegressor': XGBRegressor
        }
    
    len_dict = {
        'securities': len(securities),
        'vol_proxies': len(vol_proxies),
        'lags': len(lags)
        }
    
    counter_dict = {
        'securities': 0,
        'vol_proxies': 0,
        'lags': 0
        }
    
# =============================================================================
# instantiate the class
# =============================================================================
    gridsearch = CustomGridSearch(model_class = model_dict[model_class], 
                                  param_grid = param_grid, 
                                  min_size = min_size, 
                                  h = h,
                                  keep_forecast = keep_forecast)
# =============================================================================
    df_var = df_var.copy()
# =============================================================================
# Filter the data set to keep only the given security and volatility
# =============================================================================
    exceptions = []
    begin = time.time()
    total_iterations = len(securities) * len(vol_proxies) * len(lags)

    it = 0
    for security in securities:
        counter_dict['securities'] += 1
        counter_dict['vol_proxies'] = 0
        for vol_proxy in vol_proxies:
            counter_dict['vol_proxies'] += 1
            counter_dict['lags'] = 0
            for lag in lags:
    
                counter_dict['lags'] += 1
                
                msg = " ".join([
                    'Security: {} of {}'.format(counter_dict['securities'],
                                                len_dict['securities']),
                    'Vol Proxy: {} of {}'.format(counter_dict['vol_proxies'],
                                                 len_dict['vol_proxies']),
                    'Lags: {} of {}'.format(counter_dict['lags'],
                                            len_dict['lags'])
                    ])
                
# =============================================================================
# Check if model combination should be skipped
# =============================================================================
                
                if skip_fitted:
                    
                    check_str = (
                        vol_proxy + '_' + str(lag) + '_' + security
                        ).lower() 

                    if check_str in df_fitted['vol_lags_security'].tolist():
                        
                        continue
         
                

                df_fit = (
                    df_var[df_var['Security'].str.lower() == security.lower()]
                        .copy()    
                    )
 
    
                
# =============================================================================
# Fit the model                
# =============================================================================
                start_time = time.time()
                assert df_fit.empty == False, " ".join([
                    'Data Frame is empty after filtering for security'
                    ])
                
                try:
                    print("-"*80)
                    print(msg)
                    result = gridsearch.fit_model(
                        df = df_fit,
                        y_col = [vol_proxy],
                        x_col = x_covariates,
                        lags = lag,
                        use_standardscaler = use_standardscaler,
                        window_size = window_size,
                        log_to_level = True,
                        df_forecasts_x = df_forecasts_x
                    )
                    
                    end_time = time.time()
            
                    total_time = round((end_time - start_time) / (60 * 60), 2)
                
# =============================================================================
#  the result variable contains a dictionary, with i) the best params
#  ii) the value of the lowest MSE for both 1-step ahead forecast and
#  the lowest MSE for the overall forecast (overall is a weighted average
#  of the 1,...,5 steps ahead forecasts MSE), iii) a data frame with the 1-step
#  ahead and overall MSE for each set of params. Notice also that the target
#  variable is for the ML in log. But we transformed them again using .exp()
#  before the computation of the MSE.
#
#  We will obtain the data frame containig the parameters and MSE and add
#  further specifications to it and then append it to the existing table
#  in the database.
# =============================================================================
                
                    df_specification = result['specification'].copy()
                    df_specification['Security'] = security.upper()
                    df_specification['Model'] = model_class.upper()
                    df_specification['Vol Proxy'] = vol_proxy.title()
                    df_specification['lags'] = lag
                    df_specification['time_in_h'] = total_time

# =============================================================================
# Obtain the index of the specification with the lowest one step and overall
# MSE. Assign 1 to indicate it is the best model, and 0 otherwise.
# =============================================================================
                    df_specification['best-one-step'] = (
                        np.where(
                            df_specification['one-step'] == df_specification['one-step'].min(),
                            1,
                            0
                            )
                        )
                    
                    df_specification['best-overall'] = (
                        np.where(
                            df_specification['overall'] == df_specification['overall'].min(),
                            1,
                            0
                            )
                        )

# =============================================================================
# if exogneous covariate are passed as a list, we save them in a column
# =============================================================================
                    df_specification['Covariate'] = (
                        'None' if  x_covariates is None else str(x_covariates)
                        )
                    df_specification['Scaler'] = 'Yes' if use_standardscaler else 'No'
                    
# =============================================================================
# Export the dataframe to the database
# =============================================================================
                    if db_engine is not None:
                        df_specification.to_sql(
                            name = table_name,
                            index = False,
                            con = db_engine,
                            if_exists = 'append'
                            )
                    
             
                    print(f'{it} of {total_iterations} completed.')
                    print("-"*80)
                except:
                
                    exceptions = [[vol_proxy, lag, security, model_class]]
                    
                    columns = ['Vol Proxy', 'lag', 'Security', 'Model']
                    df_exception = pd.DataFrame(data = exceptions, 
                                                columns = columns)
                    
# =============================================================================
# If exception happens, we export it to the same db, with the same table name,
# but append the suffix_exception to the name.
# =============================================================================
                    if db_engine is not None:
                        df_exception.to_sql(
                            name = table_name + '_exception',
                            index = False,
                            con = db_engine,
                            if_exists = 'append'
                            )
                        
                    

                finally:
                    it += 1
                    
                    end_time = time.time()
                    total_time = round((start_time - end_time) / (60 * 60), 2)
                    print(f'{it} of {total_iterations} completed.')
                    print('Total time iteration in hours: {}'.format(total_time))
                    print("-"*80)
                    if db_engine is None:
                            if keep_forecast and df_exception.empty:
                                 
                                return df_specification, result['forecasts_list']
                            
                            
                            return df_specification if df_exception.empty else df_exception
                    
    end = time.time()
    time_passed = round((end - begin) / (60 * 60), 2)
    print("-"*80)
    print('Total time in (all iterations) hours: {}'.format(time_passed))

    
