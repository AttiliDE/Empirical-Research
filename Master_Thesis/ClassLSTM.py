, # -*- coding: utf-8 -*-
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

devices = tf.config.experimental.list_physical_devices('GPU')

if len(devices) > 0:
    tf.config.experimental.set_visible_devices([], 'GPU')



#%% Section 1: Class Definition
#%%
class LSTMModel():
    
    def __init__(self, df, random_seed = None):
                 
        
        self.df_raw = df.copy()
        
        self.history = []
        self.random_seed = random_seed

            
    def build_model(self, 
                    input_shape,
                    output_shape,
                    layers,
                    model_name = None,
                    dropout = 0.2, 
                    optimizer = 'adam',
                    loss: list = ['mse'],
                    metrics: list = ['mse'],
                    activation: str = 'linear',
                    add_dropout = True):

# =============================================================================
# set random seed if passed to the class as argument
# =============================================================================
        if self.random_seed:
  
            tf.keras.utils.set_random_seed(self.random_seed)
            tf.config.experimental.enable_op_determinism()
# =============================================================================
# Instantiate a sequential model.
# =============================================================================
        self.model = None
        self.model = Sequential(name = model_name)
        
        self.model.add(Input(input_shape))

# =============================================================================
# Add layers dynamically to the model. We return the sequence for each layer
# that is not the last and add a dropout to avoid overfitting (if applicable)
# =============================================================================

        for i, units in enumerate(layers, start = 1):
            
            self.model.add(
                LSTM(units, return_sequences = (i != len(layers)))
                )
                
            if add_dropout:
                self.model.add(Dropout(dropout))
# =============================================================================
# Add the output layer to the model. It should contain as many observations
# as the number of step aheads we would like to forecast. (Here 5)   
# =============================================================================
        self.model.add(Dense(output_shape, activation = activation))
        
        self.model.compile(optimizer = optimizer,
                           loss = loss,
                           metrics = metrics)
        
    def train_model(self,
                    train_X,
                    train_y,
                    batch_size = 32,
                    epochs = 10,
                    validation_data = None,
                    save_history = True,
                    verbose = 1,
                    inverse_log = True):
        
        if self.model is None:
            raise ValueError('The model has not been built yet.')
            
        self.history = self.model.fit(train_X,
                                      train_y,
                                      batch_size = batch_size,
                                      epochs = epochs,
                                      validation_data = validation_data,
                                      verbose = verbose,
                                      )
        
        if save_history:
# =============================================================================
# Create a data frame to save the training history.
# =============================================================================
            history_df = pd.DataFrame(self.history.history)
            history_df['epoch'] = history_df.index + 1
# =============================================================================
# Create a data frame to save the model architecture
# =============================================================================           
            config_df = pd.DataFrame({
                'model_name': [self.model.name],
                'architecture': [json.dumps(self.model.get_config())],
                'input_shape': [train_X.shape],
                'output_shape': [train_y.shape[1]],
                'loss_function': [self.model.loss],
                'optimizer': [self.model.optimizer.get_config()],
                'epochs': [epochs],
                'batch_size': [batch_size]
                })

# =============================================================================
# combine and return both data frames above.
# =============================================================================
            combined_df = pd.concat([config_df] * len(history_df), ignore_index = True)
            combined_df = pd.concat([combined_df, history_df], axis = 1)
           
            if validation_data:
                val_X, val_y = validation_data
                mse_original = self.calculate_original_scale_mse(val_X, val_y, inverse_log = inverse_log)
                combined_df['val_loss_original_scale_last_epoch'] = mse_original
            return combined_df
        
    def prepare_input_data(self,
                           df,
                           lags,
                           target_variable,
                           h,
                           val_size,
                           covariate_contemporaneous = True,
                           use_standardscaler = True):
        
        
        

# =============================================================================
# create list to store the formatted data        
# =============================================================================
        train_X = []
        train_y = []
        
        df = df.copy()
        
# =============================================================================
# ensure that the target variable is in the first column
# =============================================================================
        covariates = [col for col in df.columns if col != target_variable]

        cols = [target_variable] + covariates if covariates else [target_variable]
  
        df = df[cols].copy()
        
# =============================================================================
# Notice that different to the SVR and XGBoost models, the LSTM model requires
# the y_t and the x_t to have all the same dimensions. Hence, we will
# be using the lagged values instead of the contemporaneous values of x (sent
# index.)

# To make it clear the equation inside a neuron-nodel equals:
#   y_t = w_1 y_{t-1} + ... w_{p} * w_p y_{t-p} + v_1 * x_{t-1} + ... v_p * x_{t-p}
# =============================================================================

         
# =============================================================================
# Scale the data, if applicable.
# =============================================================================
        
        val_index = int(df.shape[0] * val_size)
      
        if use_standardscaler:
            
            self.scaler = StandardScaler()
            
            df_train = df.iloc[:val_index, :].copy()
     
            df_train = pd.DataFrame(
                data = self.scaler.fit_transform(df_train),
                columns = df_train.columns,
                index = df_train.index
                )
  
            df_val = df.iloc[val_index:, :].copy()

            df_val = pd.DataFrame(
                data = self.scaler.transform(df_val),
                columns = df_val.columns,
                index = df_val.index
                )
# =============================================================================
# Notice that we need the train and validation datasets to be in a single
# dataframe in order for the iteration below to work.
# =============================================================================
            df = pd.concat([df_train, df_val], axis = 0)
# =============================================================================
# obtain only the train set 
# =============================================================================
            
        for i in range(len(df) - lags - h + 1):
            
            train_X.append(df.iloc[i:i + lags, 0:df.shape[1]].values)
            
            train_y.append(df.iloc[i + lags:i + lags + h, 0].values)
            
# =============================================================================
# Create validation set from the val_size passed as argument.
# =============================================================================
                
        val_X, val_y = train_X[val_index:], train_y[val_index:]
        
        train_X, train_y = train_X[:val_index], train_y[:val_index]
        

            
# =============================================================================
# return all the data sets (test and validation)
# =============================================================================
        return (
            np.array(train_X),
            np.array(train_y),
            np.array(val_X),
            np.array(val_y)
            )
    
    
    def estimate_model(self,
                       target_variables: list,
                       securities: list,
                       lags: list,
                       h: int,
                       val_size: float,
                       layers: list,
                       batches: list,
                       epochs: list,
                       dropout:list = [0],
                       covariates = None,
                       covariate_contemporaneous = True,
                       use_standardscaler = True,
                       activation_function = 'linear',
                       combinations = True,
                       to_db = None,
                       table_name = None,
                       report_only_last_epoch = False,
                       inverse_log = True,
                       ):
    

# =============================================================================
# Assert that if a database was passed that the table name is not None.
        if to_db:
            assert table_name is not None, " ".join([
                'A table name must be passed to save the results to the database.'
                ])
# =============================================================================
# Obtain all possible combinations of LSTM-layers (units)        
# =============================================================================
        units_combinations = []

        if combinations:
            
            for i in range(len(layers)):
                units_combinations.extend(
                    list(itertools.product(layers, 
                                           repeat = len(layers) - i))
                    )
        else:
            units_combinations = [tuple(layers)]
# =============================================================================
# We now add the "dropout" values to the unit-list, this will be the last va-
# values in the tuple.       
# =============================================================================
        
        
        layers_params = list(
            itertools.product(units_combinations, dropout)
            )

        
# =============================================================================
# remove any element of the leyers_params in which the first element (the units)
# are empty
# =============================================================================
        layers_params = [
            element for element in layers_params if element[0] != ()
            ]
# =============================================================================
# create all permutation of epoch and batch sizes
# =============================================================================
        epochs_batches = list(itertools.product(epochs, batches))

# =============================================================================
# Now create all permutations of layers_params (units of the LSTM and Dropout)
# and the epochs_batches
# =============================================================================
        params = list(set(itertools.product(layers_params, epochs_batches)))

# =============================================================================
# Assert that the securities passed are in a column of the data frame.
# =============================================================================
        df_raw = self.df_raw.copy()
        
        assert set(securities).issubset(df_raw['Security'].tolist()), " ".join([
            'One or more of the securities passed is not in the "Security"-column',
            'of the dataframe.'
            ])
        
# =============================================================================
# Assert that the covariates (if applicable) and the targets are in the columns
# of the data frame.
# =============================================================================
        variables = (
            target_variables + covariates if covariates else target_variables
            )

        assert set(variables).issubset(df_raw.columns.tolist()), " ".join([
            'One or more (independent and/or) variable(s) are not in the column',
            'names of the data frame.'
            ])
# =============================================================================
# We now iterate over the list of securities, target_variables and lags.
# We then build and estimate the models.
# =============================================================================
   
        i = 0
        df_history_master = pd.DataFrame({})
        total_combinations = len(securities) * len(target_variables) * len(lags)
        
        dict_info = {}
        dict_info['securities'] = len(securities)
        dict_info['targets'] = len(target_variables)
        dict_info['lags'] = len(lags)
        
        
        start_time = datetime.now()
        print('Starting Estimation {}:{}:{}'.format(
            start_time.hour,
            start_time.minute,
            start_time.second
            ))
        print('-'*80)
        print('\n')
        
        s = 0
        j = 0
        for security in securities:
            s += 1
            t = 0
            for target_variable in target_variables:
                t += 1
                l = 0
                for lag in lags:
                    l += 1
                    j += 1
                    substart_time = datetime.now()
                    msg = " ".join([
                        'Training model {} of {},'.format(j, total_combinations),
                        'for Security {} of {},'.format(s, dict_info['securities']),

                        'Target {} of {},'.format(t, dict_info['targets']),
                        'and lag {} of {}'.format(l, dict_info['lags'])
                        ])
                    print(msg)
                    print('-'*80)
  
# =============================================================================
# obtain the data for the security in a separate dataframe.
# =============================================================================

                    df = self.df_raw[self.df_raw['Security'] == security].copy()
                
                

                    cols = [target_variable] + covariates if covariates else [target_variable]
                    
                    df = df[cols].copy()
                    
# =============================================================================
# Prepare the data used to train the model.
# =============================================================================
                    train_X, train_y, val_X, val_y = self.prepare_input_data(
                        df = df, 
                        lags = lag, 
                        target_variable = target_variable, 
                        h = h, 
                        val_size = val_size,
                        use_standardscaler = use_standardscaler,
                        covariate_contemporaneous = covariate_contemporaneous
                        )
# =============================================================================
# We now need to build the model based on the number of layers passed as argu-
# ment. And then train it given the batches and epochs.
# We will iterate over the list of params and train the model one by one.
# =============================================================================

                    for param in tqdm(
                            params, 
                            desc = 'Bulding and estimating LSTM-Models',
                            unit = 'it',
                            ncols = 100
                            ):
                        i += 1
                        model_name = f'ID_{i}'
                        print('\n')
                        print('-'*80)
                        print(msg)
                        print('-'*80)
                        print('\n')
                        self.build_model(input_shape = (train_X.shape[1], train_X.shape[2]), 
                                         output_shape = train_y.shape[1], 
                                         layers = param[0][0],
                                         dropout = param[0][1],
                                         model_name = model_name)
                        
                        
                        history_temp = self.train_model(
                            train_X = train_X,
                            train_y = train_y,
                            epochs = param[1][0],
                            batch_size = param[1][1],
                            validation_data = (val_X, val_y),
                            inverse_log = inverse_log
                            )
                        
                        
                        history_temp['security'] = security
                        history_temp['target'] = target_variable
                        history_temp['lags'] = lag
                        history_temp['stardard_scaler'] = use_standardscaler
                        history_temp['covariates'] = str(covariates)
          
                        try:
                            history_temp['layers_units'] = [param[0][0]] 
                        except:
                            history_temp['layers_units'] = [param[0][0]] * len(history_temp)
                     
                        history_temp['dropout'] = param[0][1]
                        history_temp['total-epochs'] = param[1][0]
                        history_temp['batch_size'] = param[1][1]
                        if df_history_master.empty:
                            df_history_master = history_temp
                        
                        else:
                            df_history_master = pd.concat(
                                [df_history_master, history_temp], axis = 0
                                )

                        df_history_master.reset_index(inplace = True, drop = True)
# =============================================================================
# If a databank was passed, we store the table to the database
# =============================================================================
                if to_db:
# =============================================================================
# convert all columns of the data frame to "string" in order to save it 
# to the database.
# =============================================================================
                    df_to_db = df_history_master.copy()
                    
                    df_to_db = df_to_db.astype(str)
                        
                    df_to_db.to_sql(
                        name = table_name,
                        index = False,
                        if_exists = 'append',
                        con = to_db
                        )
                    
                    df_history_master = pd.DataFrame({})
                
                
                subend_time = datetime.now()
                time_elapsed = subend_time - substart_time
                print(f'\n\nTime elapsed (iteration): {round((time_elapsed.total_seconds()) / (60 * 60),3)} h.')
        
        
        end_time = datetime.now()
        time_elapsed = end_time - start_time
        print('-'*80)
        print(f'Total time elapsed: {round((time_elapsed.total_seconds()) / (60 * 60), 3)} h.')

# =============================================================================
# If the results are not stored to a database, we return the dataframe
# =============================================================================
        if not to_db:

# =============================================================================
# If report only last epoch is true, we return only the results for the last epoch (this is relevant if we would like
# to re-train the best model)
# =============================================================================
            if report_only_last_epoch:
                df_history_master = df_history_master[df_history_master['epoch'] == df_history_master['total-epochs']]
            return df_history_master

    def calculate_original_scale_mse(self, val_X, val_y, inverse_log = True):
        """
        Calculate MSE on the original scale by reversing the scaling 
        transformation

        Parameters
        ----------
        val_X : np.array
            Validation feature data.
        val_Y : np.array
            Target variable

        Returns
        -------
        MSE calculated on the orginal scale.

        """
        
# =============================================================================
# Make sure that the model was built and that a scaler is available
# =============================================================================
        if self.model is None:
            
            raise ValueError('The model has not been built yet.')
            
        if self.scaler is None:
            raise ValueError('Scaler not avaiable, unable to inverse transform.')
            
# =============================================================================
# Generate predictions on the scaled validation data
# =============================================================================
        predictions_scaled = self.model.predict(val_X)

# =============================================================================
# Reverse the predictions using the scaler
# =============================================================================
        predictions_original = self.scaler.inverse_transform(predictions_scaled)
        
        val_y_original = self.scaler.inverse_transform(val_y)

# =============================================================================
# If inverse-log equal to true, we apply the exponentatial function to the val_y_original and predictions
# =============================================================================
        
        if inverse_log:
            val_y_original = np.exp(val_y_original)
            predictions_original = np.exp(predictions_original)

# =============================================================================
# Compute mse
# =============================================================================
        mse_original = np.mean(np.square(predictions_original - val_y_original))
        return mse_original
    def predict(self, df:pd.DataFrame):
        """
        Function to predict values using the data in df.

        Parameters
        ----------
        df : TYPE
            DESCRIPTION.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        if self.model is None:
            
            raise ValueError('The model has not been built yet.')
            
# =============================================================================
# Make predictions using the trained model
# =============================================================================
        self.model.predict(df)
        

