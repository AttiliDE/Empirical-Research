#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 22:19:55 2024

@author: ferreira
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 17:53:37 2024

@author: ferreira
"""

#%% Section 0: Imports
#%%
import pandas as pd
import numpy as np
from sqlalchemy.engine import Engine, create_engine
from itertools import product
from typing import Union, List, Tuple, Callable, Generator
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from arch import arch_model
import os
from functools import partial
from scipy.optimize import OptimizeWarning
import warnings
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score
    )
from arch.univariate import (
    GARCH,
    EGARCH,
    FIGARCH,
    ARCH,
    ConstantMean,
    ZeroMean,
    Normal,
    StudentsT,
    SkewStudent,
    GeneralizedError
    )

import re



#%% Section 1: Class for GARCH-Model Estimation
#%%

class GarchModels():

    def __init__(self,
                 df: pd.DataFrame,
                 mean: Union[List[str], str] = 'all', 
                 vol: Union[List[str], str]= 'all', 
                 dist: Union[List[str], str] =  'all',
                 max_comb: int = 50, db:Engine = None):

        """
        

        Parameters
        ----------
        df : pd.DataFrame
            A dataframe containing a single column with the returns in its-
            column and date-time formated date as its indices.
        mean : list | str, optional
            mean dynamics to be used in the mean equation of the GARCH family 
            of models.
            Accepted arguments:
                - 'Constant'
                - 'Zero'
                - 'all'
            Any other argument than all needs to be passed inside a list
            If all is selected, a model is estimated for all mean dynamics.
            The default is 'all'.
        dist : str, optional
            The distribution of the innovation eta in the model.
            Accepted arguments:
                - 'normal'
                - 't'
                - 'skewt'
                - 'ged'
                - 'all'
            The default is 'all'.
        max_comb : int, optional
            The total number of p, q, and o combinations for interative estimation. 
            The default is 50.
        db : Engine, optional
            An engine pointing to the database where the results will be saved. 
            The default is None.

        Returns
        -------
        None.

        """
# =============================================================================
# Assert that the data was passed as a pandas dataframe and that it has at most
# one column. And apply the function __format_strings() to make sure that the
# arguments were passed with lower/upper/proper cases.
# =============================================================================
        df = df.to_frame() if isinstance(df, pd.Series) else df
        
        assert isinstance(df, pd.DataFrame) and df.shape[1] == 1, " ".join([
            "Class expects a single column pandas-dataframe with the returns",
            "used in the estimation."
            ])
        
        self.df = df
        
# =============================================================================
# Assert that the index column of the dataframe is in date format       
# =============================================================================
        assert isinstance(df.index, pd.DatetimeIndex), " ".join([
            'The index column of the dataframe containing the returns',
            'must be passed in pd.DatetimeIndex format.'
            ])
        
        mean_list = ['Zero', 'Constant']
        dist_list = ['normal', 't', 'skewt', 'ged']
        vol_list = ['ARCH', 'GARCH', 'EGARCH', 'FIGARCH']
        cv_types = ['origin', 'sliding']
        loss_funcs = ['mse', 'msle', 'mae', 'mape', 'r2']
        
        self.mean = mean_list if mean == 'all' else mean
        self.dist = dist_list if dist == 'all' else dist
        self.vol = vol_list if vol == 'all' else vol
        self.max_comb = max_comb
        self.cv_types = cv_types
        self.loss_funcs = loss_funcs
        self.engine = db
        
# =============================================================================
# Assert that the values passed are valid
# =============================================================================
        assert (
            isinstance(self.mean, list) and
            isinstance(self.dist, list) and
            isinstance(self.vol, list)
            ), " ".join([
            'mean, vol and dist arguments must be either "all" or passed as a',
            'list of strings.'
            ])

            
        if mean != 'all':
            
            self.mean = self.__format_strings(self.mean, format_type = 'title')
            
            assert len(set(self.mean).intersect(mean_list)) == len(self.mean), " ".join([
                'One or more elements in the mean-list is (are) not valid.', 
                'Valid arguments are "Zero" and "Constant".'
                ])
            
        if dist != 'all':
            
            self.dist = self.__format_strings(self.dist, format_type = 'lower')
            
            assert len(set(self.dist).interset(dist_list)) == len(self.mean), " ".join([
                'One or more elements in the dist-lit is (are) not valid.',
                'Valid arguments are "normal", "t", "skewt", and "ged".'
                ])
            
        if vol != 'all':
            
            self.vol = self.__format_strings(self.vol, format_type = 'upper')
            
            assert len(set(self.vol).intersect(vol_list) == len(self.vol)), " ".join([
                'One or more elements in the vol-list is (are) not valid.',
                'Valid arguments are "ARCH", "GARCH", "EGARCH", "FIGARCH".'
                ])

# =============================================================================
# Call the function to create data-frame with combinations of arguments to 
# be passed to arch_model function.    
# =============================================================================

        self.df_args = self.__create_model_args()
        
        self.df_args.sort_values(by = ['vol', 'mean', 'dist', 'p', 'q', 'o'],
                                inplace = True)
    
    def __format_strings(self, list_strings:list, format_type):
        """
        Function formats strings in lower, upper and proper (title) cases.        
        
        Parameters
        ----------
        list_strings : list
            A list containing the strings to be formatted.
        format_type : TYPE
            The type of formatting. Valid arguments:
                - lower: lower cases all strings.
                - upper: upper cases all strings.
                - title: upper cases first letter and lower cases all others.

        Returns
        -------
        A list with the formatted strings.

        """
        if format_type == 'lower':
            formatted_strings = [string.lower() for string in list_strings]
        
        elif format_type == 'upper':
            formatted_strings = [string.upper() for string in list_strings]
            
        elif format_type == 'title':
            formatted_strings = [string.title() for string in list_strings]
            
            
        return formatted_strings
    
    def __create_model_args(self):
        """
        Function to create a dataframe with parameters combinations to  be used 
        in interative procedure.
        
        Returns
        -------
        A pd.DataFrame containing all possible combinations of (p, q, o), vol,
        mean, and dist passed as arguments to the class.

        """
# =============================================================================
# Start a data-frame with all combinations for 0 to max_comb for the values
# "p", "q", and "o" used as arguments in the arch_model()-call.
# =============================================================================
        df_comb = pd.DataFrame(
            columns = ['p', 'q', 'o'],
            data = product([*range(0, self.max_comb + 1)], repeat = 3)
            )
    
# =============================================================================
# A dataframe will be initialized to stored those repetitions for each of the
# values passed as a list in the mean dynamics and distribution at instantiati-
# on. (see __init__)    
# =============================================================================
        df_args = pd.DataFrame(
            columns = ['mean', 'vol', 'p', 'q', 'o', 'dist']
            )
        
# =============================================================================
# Start the procedure of creating a p,q,o combination for each combination of 
# dist, mean, and vol.
# =============================================================================
        for mean in tqdm(self.mean, desc = 'Preparing model arguments',
                         ncols = 100):
                
            for vol in self.vol:
                
                for dist in self.dist:

# =============================================================================
# Copy the df_args dataframe and extend the copy with columns to store the mean
# the volatility and the distributional parameters.
# Once the expansion is done, the temporary dataframe will be appended to the
# df_args dataframe. At the end the df_args dataframe will contain all possible
# combinations of the parameters.
# =============================================================================                    
                    df_args_temp = df_comb.copy(deep = True)
                    
                    df_args_temp['mean'] = mean
                    df_args_temp['vol'] = vol
                    df_args_temp['dist'] = dist

                    df_args = pd.concat([df_args, df_args_temp], axis = 0)

# =============================================================================
# The next step consists of cleaning the df_args based on the following argu-
# ments.

# 1. FIGARCH orders accept at most p = q = 1. All values above 1 are useless.
# Further, the model does not take "o" parameter; any combination of values
# with o above 0 can also be removed.

# 2. ARCH models are based on the parameter p. Any q and o greater 0 are use-
# less and any p = 0 can also be removed.

# 3. GARCH model accepts at minimum a value of (p, q) = (1, 0). Hence, all 
# p = 0 can be removed.

# 4. EGARCH demands at least one of the parameters (p, o) to be non-zero.
# It follows that we can exclude combinations of p = o = 0. 
# =============================================================================

        filt1 = (
            (df_args['vol'] == 'FIGARCH') &
            ((df_args['p'] > 1) | (df_args['q'] > 1) | (df_args['o'] > 0))
            )
        
        df_args = df_args[~filt1]
        
        filt2 = (
            (df_args['vol'] == 'ARCH') &
            ((df_args['q'] > 0) | (df_args['o'] > 0)) |
            (df_args['p'] == 0)
            )
        
        df_args = df_args[~filt2]
        
        filt3 = (
            (df_args['vol'] == 'GARCH') & (df_args['p'] == 0)
            )
        
        df_args = df_args[~filt3]
        
        filt4 = (
            (df_args['vol'] == 'EGARCH') & 
            ((df_args['p'] == 0) & (df_args['o'] == 0))
            )
        
        df_args = df_args[~filt4]

        
        df_args = df_args[~filt4]
        df_args.index = range(1, df_args.shape[0] + 1)
        
        return df_args
    
    def __check_if_args_are_valid(self, dict_args:dict):
        
        """
        Function to check whether the arguments to create arch_model are valid
        arguments stored as attribute to the cass.

        Parameters
        ----------
        dict_args : dict
            A dictionary containing the arguments to be checked as its keys
            and a list (or a tuple) with the arguments to be checked as their 
            values.

        Returns
        -------
        None or an assertion-error if the requirements are not met.

        """
        
        
        for key, values in dict_args.items():
            
            if key == 'vol':
                            
                assert all([value in self.vol for value in values]), " ".join([
                    'One or more of the volatility model(s) passed is (are)',
                    'not valid.'
                    ])
                
            if key == 'mean':
           
                assert all([value in self.mean for value in values]), " ".join([
                    'One or more of the mean model(s) passed is (are)',
                    'not valid.'
                    ])
                
            if key == 'dist':
                
                assert isinstance(values, list), " ".join([
                    'dist argument must be passed as a list.'
                    ])
                
                assert all([value in self.dist for value in values]), " ".join([
                    'One or more of the distribution(s) passed is (are)',
                    'not valid.'
                    ])
                
            if key == 'pqo':
              
                assert len(values) == 3, " ".join([
                    'pqo must be a tuple containing three integers as the',
                    'max (if applicable) values to try for (p, q, o).'
                    ])
                assert max(values) > 0, " ".join([
                    'At least one of the arguments (p, q, o) must be non-zero.'
                    ])
            
            if key == 'cv_type':
                
                assert values in self.cv_types, " ".join(
                    ['Value passed to cv_types is not valid. Valid arguments',
                     'are one of the following: "origin" and "sliding".']
                    )
                
    def __assert_instance(self, args, inst, arg_name = ""):
        """
        Function to assert that the argument passed is of a given instance.

        Parameters
        ----------
        args : TYPE
            The argument that will be checked.
        inst : TYPE
            The instance to check agains.
        arg_name : TYPE, optional
            Name of the argument

        Returns
        -------
        An AssertionError if instance does not match.

        """
        
        assert isinstance(args, inst), " ".join([
            f'{arg_name} must be passed as a {str(inst)}'
            ])


    def __specify_arch_model():
        pass
    
    def __query_specification(self,
                               mean_list:list,
                               vol_list:list,
                               pqo_tuple:tuple,
                               dist_list:list,
                               pqo_ismax:bool = False):
        """
        Function to obtain the indices of the rows in the df_args meeting
        the requirements passed as arguments.

        Parameters
        ----------
        mean_list : list
            A list of mean specifications.
        vol_list : list
            A list of volatility specifications.
        pqo_tuple : tuple
            A tuple of integer indicating the values (p, q, o)
        dist_list : list
            A list of strings with the distributions.
        pqo_ismax : bool, optional
            Flag (boolean) indicating whether the values passed to pqo_tuple 
            are max-values or exact values.
            The default is False.

        Returns
        -------
        None.

        """

# =============================================================================
# Query first the pqo-values to reduce the dimensionality of the data-frame
# df_args stored as attribute to the class. In a later step, obtain only rows
# whose values are in the lists passed as arguments
# =============================================================================

        if pqo_ismax:
            
            filt_pqo = (
                (self.df_args['p'] <= pqo_tuple[0]) &
                (self.df_args['q'] <= pqo_tuple[1]) &
                (self.df_args['o'] <= pqo_tuple[2])
                )
        
        else:
            
            filt_pqo = (
                (self.df_args['p'] == pqo_tuple[0]) &
                (self.df_args['q'] == pqo_tuple[1]) &
                (self.df_args['o'] == pqo_tuple[2]) 
                )
            
            
        df_temp = self.df_args[filt_pqo]
      
        filt = (
            (df_temp['mean'].isin(mean_list)) &
            (df_temp['vol'].isin(vol_list)) &
            (df_temp['dist'].isin(dist_list))
            )
        
       
        queried_indices = df_temp[filt].index.tolist()
        
        return queried_indices
    
    @staticmethod
    def compute_ic(params_dict:dict, df: pd.DataFrame, ic:str):
        """
        Function to pass into the the parallelization of 
        compute_robust_infocriterion()

        Parameters
        ----------
        params_dict : dict
            A dictionary containing the parameters to estimate the model and 
            the model ID
        df : pd.DataFrame
            A dataframe containing the data to estimate the models.
        ic : str
            The information criterion to be used. Valid arguments:
                - 'bic': computes Bayesian Information Criterion.
                - 'aic': computed Akaike Information Criterion.

        Returns
        -------
        A dictionary containing the heteroscedastic adjusted and the standard
        values for the information criterion passed as arguments as well as the
        model ID.
        """
        
        
        
# =============================================================================
# Fit the model according to the parameters passed, obtain conditional
# volatility and square it to obtain variance. Finally, compute the HSIC
# accoding to formula provided by Brooks and Burke (2003: 558).

# Notice that the params_dict contain an ID-key to uniquely identify the model.
# These keys are equivalent to the indices of the df_args.
# =============================================================================

        try:
            seed = params_dict.pop('seed')
        except:
            seed = None
            
        mean_dict = {
            'Zero': ZeroMean,
            'Constant': ConstantMean
            }
        
        vol_dict = {
            'ARCH': ARCH,
            'GARCH': GARCH,
            'EGARCH': EGARCH,
            'FIGARCH': FIGARCH
            }
        
        np.random.seed(seed)
        dist_dict = {
            'normal': Normal(seed = seed),
            't': StudentsT(seed = seed),
            'skewt': SkewStudent(seed = seed),
            'ged': GeneralizedError(seed = seed)
            }
        
        if params_dict['vol'] == "GARCH" or params_dict['vol'] == "EGARCH":
            
            vol_model = vol_dict[params_dict['vol']](
                p = int(params_dict['p']),
                q = int(params_dict['q']),
                o = int(params_dict['o'])
                )
            
        elif params_dict['vol'] == 'ARCH':

            vol_model = vol_dict[params_dict['vol']](
                p = int(params_dict['p']),
                )

            
        elif params_dict['vol'] == 'FIGARCH':
            
            vol_model = vol_dict[params_dict['vol']](
                p = int(params_dict['p']),
                q = int(params_dict['q'])
             
                )
            
                           
        model_id = params_dict.pop('ID')
        
        warnings.warn = lambda *a, **kw: False
        model = mean_dict[params_dict['mean']](
            df,
            volatility = vol_model,
            distribution = dist_dict[params_dict['dist']]
            )
            
            
# =============================================================================
# Now fit the model, generate forecasts, compute the values of the loss func-
# tions, save the parameters of each iteration and the model specifications
# in a separate data frame. Aggregate all dataframes in a dictionary and return
# it.
# =============================================================================

        fitted_model = model.fit(disp = False, options = {'maxiter': 10000})
    
        
        def HAIC(cond_vol, num_params):
            """
            Function to compute the heteroscedastic adjusted AKAIKE information
            criterion value.

            Parameters
            ----------
            cond_vol : TYPE
                A numpy array with the conditional volality of a (G)arch-model
                estimated by the arch_model() package.
            num_params : TYPE
                The total number of parameters estimated in the model

            Returns
            -------
            The value of HAIC.

            """
            
            sum_log_squared_var = np.sum(np.log(np.power(cond_vol, 2)))
            HAIC_val = sum_log_squared_var + 2 * num_params
            
            return HAIC_val
            
        def HSIC(cond_vol, num_params):
            """
            Function to compute the heteroscedastic adjusted Bayesian informa-
            tion criterion value.

            Parameters
            ----------
            cond_vol : TYPE
                A numpy array with the conditional volality of a (G)arch-model
                estimated by the arch_model() package.
            num_params : TYPE
                The total number of parameters estimated in the model

            Returns
            -------
            The value of HSIC

            """
            
            sum_log_squared_var = np.sum(np.log(np.power(cond_vol, 2)))
            HSIC_val = sum_log_squared_var + num_params * np.log(cond_vol.shape[0])
            
            return HSIC_val


        result_dict = {}
# =============================================================================
# Check whether the model converged and continue with the computation only in that case
# =============================================================================
        if fitted_model.optimization_result['success'] == True:
            conditional_volatility = fitted_model.conditional_volatility
            num_params = fitted_model.num_params
            
            
            if ic == 'bic':
                
                result_dict['HBIC'] = HSIC(
                    cond_vol = conditional_volatility,
                    num_params = fitted_model.num_params
                    )
                
                result_dict['BIC'] = fitted_model.bic
                
            elif ic == 'aic':
                
                result_dict['HAIC'] = HAIC(
                    cond_vol = conditional_volatility,
                    num_params = num_params
                    )
                
                result_dict['AIC'] = fitted_model.aic
                
            elif ic == 'all':
                
                result_dict['HAIC'] = HAIC(
                    cond_vol = conditional_volatility,
                    num_params = num_params
                    )
                
                result_dict['AIC'] = fitted_model.aic
    
                result_dict['HBIC'] = HSIC(
                    cond_vol = conditional_volatility,
                    num_params = fitted_model.num_params
                    )
                
                result_dict['BIC'] = fitted_model.bic
        else: 

            if ic == 'bic':
                
                result_dict['HBIC'] = 100_000_000
                
                result_dict['BIC'] = 100_000_000
                
            elif ic == 'aic':
                
                result_dict['HAIC'] = 100_000_000
                
                result_dict['AIC'] = 100_000_000
                
            elif ic == 'all':
                
                result_dict['HAIC'] = 100_000_000
                
                result_dict['AIC'] = 100_000_000
    
                result_dict['HBIC'] = 100_000_000
                
                result_dict['BIC'] = 100_000_000
            
        result_dict['ID'] = model_id
        
        return result_dict
    
    def compute_robust_infocriterion(self, 
                                     ic:str = 'bic',
                                     max_workers:int = os.cpu_count(),
                                     mean:Union[List[str], None] = None,
                                     vol: Union[List[str], None] = None,
                                     pqo: Union[Tuple[int], None] = None,
                                     pqo_ismax: bool = True,
                                     dist: Union[List[str], None] = None,
                                     seed: int = None):
                                     
        """
        Function to compute conditional heteroscedastic robust information
        criterion for the set of specifications passed as argument.

        Parameters
        ----------
        ic : str, optional
            Specify which information criterion to implement. Valid arguments:
                - 'aic': het. adj. Akaike Informaiton Criterion
                - 'bic': het. adj. Schwarz / Bayesian Information Criterion
                - 'all': both information criteria as defined above.
            The default is 'bic'.
        max_workers: int, optional
            Specify the number of cpus to use with concurrent.futures.
            The default is os.cpu_count() [i.e. all cpus.]
        mean : Union[List[str], None], optional
            A list of mean models to use in the estimation of the conditional
            volatility used in the computation of the info criterion.
            If None is passed, all models stored in the class will be used.
            The default is None.
        vol : Union[List[str], None], optional
            A list of volatility models to use in the estimation of the condi-
            tional volatility used in the computation of the info criterion.
            If None is passed, all models stored in the class will be used.
            The default is None.
        pqo : Union[Tuple[int], None], optional
            A tuple of integers (p, q, o) specifying the maximum values of p, 
            q, and o to pass to the volatility model. If a p, q, and o are 
            specified above the ones in the class, the max value stored in the
            class will be used. 
            If None is passed, then the maximum value stored in the class will
            be used (p = q = o = 50).
            The default is None.
        pqo_ismax : bool, optional
            Flag indicating whether the values passed to the pqo-argument are
            the maximum (True) or exact values (False).
            The default is True.
        dist : Union[List[str], None], optional
            A list of distribution to use in the estimation of the conditional
            volatility used in the computation of the info criterion.
            If None is passed, all distributions stored in the class will be used.
            The default is None.
        seed: A random seed to pass to the model to ensure reproducibility.
        Returns
        -------
        None.

        """
        
# =============================================================================
# Start by checking if arguments passed to the function are valid and by
# formatting the strings transforming them to lower, upper or proper cases.
# =============================================================================
        if mean is not None:
            
            self.__assert_instance(args = mean, arg_name = 'mean', inst = list)
            
            mean = self.__format_strings(
                list_strings = mean, format_type = 'title'
                )
            
            self.__check_if_args_are_valid(
                dict_args = dict({'mean': mean})
                )
        else:
            
            mean = self.mean
            
        if vol is not None:
            
            self.__assert_instance(args = vol, arg_name = 'vol', inst = list)
            
            vol = self.__format_strings(
                list_strings = vol, format_type = 'upper'
                )
            
            self.__check_if_args_are_valid(
                dict_args = dict({'vol': vol})
                )
        
        else:
            
            vol = self.vol
            
        if pqo is not None:
    
            self.__assert_instance(args = pqo, arg_name = "pqo", inst = tuple)
            self.__check_if_args_are_valid(
                dict_args = dict({'pqo': list(pqo)})
                )
        else:
            
            pqo = (self.max_comb, self.max_comb, self.max_comb)
            
        if dist is not None:
            
            self.__assert_instance(args = dist, arg_name = 'dist', inst = list)
            
            dist = self.__format_strings(
                list_strings = dist, format_type = 'lower'
                )
            
            self.__check_if_args_are_valid(
                dict_args = dict({'dist': dist})
                )

# =============================================================================
# Assert that the argument passed to the information criterion (ic) is a valid
# one.                       
# =============================================================================
        valid_ics = ['bic', 'aic', 'all']
        ic = ic.lower()
        
        self.__assert_instance(args = ic, arg_name = 'ic', inst = str)
        
        assert ic in valid_ics, " ".join([
            "ic passed is not a valid argument.",
            "Valid arguments are 'ic', 'bic' or 'all'"
            ])
        
# =============================================================================
# Query the self.df_args to obtain the indices of the rows whose values meet
# the arguments passed to the function. Once the arguments are obtained, use
# the indices to create dictionaries with parameters to estimate the models.
# =============================================================================

        queried_indices = self.__query_specification(
            mean_list = mean,
            vol_list = vol,
            dist_list = dist,
            pqo_tuple = pqo,
            pqo_ismax = pqo_ismax
            )
        
# =============================================================================
# Create a new dataframe containig the indicies queried, sort the dataframe by
# Volatility, Mean, distribution, p, q, and o. Transform the dataframe to a list
# of dictionaries, and pass them as argument to the arch_model function. This
# will return models that will be fitted in a latter step
# =============================================================================
        
        df_filtered = self.df_args.loc[queried_indices]
        
        df_filtered.sort_values(by = ['vol', 'mean', 'dist', 'p', 'q', 'o'],
                                inplace = True)
# =============================================================================
# Reindex the dataframe so that the indices are ordered from 1 to shape[0]       
# =============================================================================
        df_filtered.index = range(1, df_filtered.shape[0] + 1)
        df_filtered.rename_axis('ID', inplace = True)
        df_filtered.reset_index(inplace = True)
        
        if seed is not None:
            df_filtered['seed'] = seed
            
        processes_input = df_filtered.to_dict(orient = 'records')
    
# =============================================================================
# Parallelize the process of computation with concurrent.futures
# =============================================================================
        partial_compute_ic = partial(GarchModels.compute_ic, df = self.df, ic = ic)
        with ProcessPoolExecutor(max_workers = max_workers) as executor:

            results = list(
                tqdm(
                    executor.map(partial_compute_ic, processes_input),
                    ncols = 100,
                    desc = 'Computing het. adj. IC',
                    total = len(processes_input)
                    )
                )
    
# =============================================================================
# Create a dataframe with the results and use its 'ID'-column to merge it
# with the input-dataframe df_filtered.
# =============================================================================
        if seed is not None:
            df_filtered.drop(columns = ['seed'], inplace = True)
            
        df_output = pd.DataFrame(results)
        df_result = pd.merge(df_filtered, df_output, on = 'ID')
        
        return df_result
    
    
    @staticmethod
    def CV_score(df_return:pd.DataFrame,
                 df_var:pd.DataFrame,
                 loss_funcs:list,
                 h:int,
                 size:int,
                 seed:int,
                 cv_type:str,
                 params_dict:dict,
                 return_cvgen_only = False
                 ):
        
        warnings.filterwarnings('ignore', category = OptimizeWarning)
                
        """
        Function to use in the parallelization of Cross_validation_evaluation

        Parameters
        ----------
        df_return : pd.DataFrame
            A single column pandas dataframe containing the returns and 
            pd.DatetimeIndex dates as its index.
        df_var : pd.DataFrame
            A pandas dataframe containing the volatility proxies in its columns
            and a pd.DatetimeIndex dates as its index.
        params_dict : dict
            A dictionary containing the mean, volatility, distribution and 
            p, q, o specifications to create and fit the model.
        h: int, optional
            The horizon of the variance forecasts.
            The default is 5.
        size: int, optional
            The (min) size of the dataset to compute the model using cross va-
            lidation from the origin or the size of the sliding window.
            The default is 250.
        loss_funcs : list
            A list of loss functions used in cross-validation. Valid arguments
            are:
                - mse: mean squared error
                - msle: mean squared log error
                - mae: mean absolute error
                - mape: mean absolute percentage error
                - r2: r-squared
        seed : int, optional
            The random-state passed to the forecast. Only relevant for models
            whose forecasts are made by simulation method.
            The default is None.
        cv_type : str, optional
            The type of cross-validation to use. Valid arguments are:
                - origin: uses the rolling-forecast from origin method
                - sliding: uses the sliding window forecast method.
                . The default is 'origin'.
        return_cvgen_only: bool, optional
            Boolean value to indicate whether only the CV generator should be
            returned.
            The default  is False.
        Returns
        -------
        A dictionary containing:
            master_dict = {
                'ID': {
                    val_var: the target variance proxies,
                    val_forecast_var: the forecasted variances,
                    model_params: the estimated parameters of the model,
                    eval: the evaluation measures
                    resid_and_cvol: data frame containing residuals, standardi-
                                    zed residuals and condition variance.
                    }
                }
          each key contains a dataframe as value. In the case the optimization 
          fails to converge, the dataframes are replaced by 'Convergence Error.'
          
          A eval data frame is always returned. In the case of convergence pro-
          blem, the eval-dataframe will contain the values -100.

        """
# =============================================================================
# Create the generator for the cross validation.       
# =============================================================================
        def CV_RollingFromOrigin(df:pd.DataFrame, min_size:int, h:int):
            """
            A generator to create train and validation split
            for cross validation by using the rolling-forecasting-origin method        
            
            Nota Bene: Make sure that the data set is in ascending order regarding
            the dates.
            
            Parameters
            ----------
            df : pd.DataFrame
                A padas dataframe with the data to be used in the cross-validation.
            min_size : int
                The minimum-size of datapoints needed to estimate the model.
            h : int
                The horizon of the prediction (i.e. the size of the validation set)
    
            Returns
            -------
            A generator for train and validation splits. The data is returned as a
            one dimensional numpy array.
    
            """
                
# =============================================================================
# Make sure that the data is passed as a pandas dataframe containing a single
# column, and transform this column to a numpy array.
# =============================================================================
            assert isinstance(df, pd.DataFrame), " ".join([
                'df must be a pandas dataframe.'
                ])
            
            assert df.shape[1] == 1, " ".join(
                ['Data-Frame must contain a single column']
                )
            
            data = df.to_numpy().ravel()
# =============================================================================
# iterate over the data set and make sure to have enough data to create the va-
# lidation set, which is always h-ahead of the train set.
# =============================================================================
            for period in range(len(data) - min_size - h + 1):
# =============================================================================
# The data is added sequentially and the validation split moves with period.
# =============================================================================
                train_split = data[ :min_size + period]
                val_split = data[min_size + period: min_size + period + h]
                
                yield train_split, val_split
                
                
        def CV_SlidingWindow(df:pd.DataFrame, window: int, h:int):
            """
            A generator to create train and validation split for cross validion
            by using the sliding window forecasting method.
            
            Nota Bene: Make sure that the data set is in ascending order regarding
            the dates.
    
            Parameters
            ----------
            df : pd.DataFrame
                A padas dataframe with the data to be used in the cross-validation.
            window : int
                The window-size of datapoints used to estimate the model. Notice
                that this window must be equal or larger than the minimum size of
                data points needed to estimate the model.
            h : int
                The horizon of the prediction (i.e. the size of the validation set)
    
            Returns
            -------
            A generator for train and validation splits. The data is returned as a
            one dimensional numpy array.
    
            """
    
# =============================================================================
# Make sure that the data is passed as a pandas dataframe containing a single
# column, and transform this column to a numpy array.
# =============================================================================
            assert isinstance(df, pd.DataFrame), " ".join([
                'df must be passed as a pandas dataframe.'
                ])
            
            assert df.shape[1] == 1, " ".join(
                ['Data-Frame must contain a single column']
                )
            
            data = df.to_numpy().ravel()
            
# =============================================================================
# iterate over the dataset and make sure to have enough data to create the va-
# lidation set, which is always h-ahead of the train set.
# =============================================================================
            for period in range(len(data) - window - h + 1):
# =============================================================================
# make sure that the window slides each period by one data point.
# =============================================================================
                train_split = data[period:window + period]
                val_split = data[window:window + period + h]
                
                yield train_split, val_split
                
                
# =============================================================================
# Create a dictionary with the cv-generator functions, so that the type of 
# cross validation can be chosen dynamically.
# =============================================================================
        dict_gen = {
            'origin': CV_RollingFromOrigin,
            'sliding': CV_SlidingWindow
            }

# =============================================================================
# create CV generator
# =============================================================================
        cv_gen = dict_gen[cv_type](df_return.index.to_frame(), size, h)
        
# =============================================================================
# If return_cvgen_only is set to true, we return the generator.
# =============================================================================
        if return_cvgen_only:
            return cv_gen
# =============================================================================
# Nota Bene: The models need to be constructed using their objects counterparts
# as for the EGARCH case, there is no analytically solution for the forecast.
# There are two ways to obtain forecasts in these cases, the first is by using 
# a simulation method, where the errors are drawn from the distribution passed
# to the model. The second is by bootstrapping. Here the simulated cased will
# be applied.
# =============================================================================
        mean_dict = {
            'Zero': ZeroMean,
            'Constant': ConstantMean
            }
        
        vol_dict = {
            'ARCH': ARCH,
            'GARCH': GARCH,
            'EGARCH': EGARCH,
            'FIGARCH': FIGARCH
            }
        
        np.random.seed(seed)
        dist_dict = {
            'normal': Normal(seed = seed),
            't': StudentsT(seed = seed),
            'skewt': SkewStudent(seed = seed),
            'ged': GeneralizedError(seed = seed)
            }

        
        
        forecast_method = (
            'simulation' if params_dict['vol'] == 'EGARCH' else 'analytic'
            ) 
        
        
        loss_functions_dict = {
            'mae': mean_absolute_error,
            'mse': mean_squared_error,
            'r2': r2_score,
            'mape': mean_absolute_percentage_error,
            'msle': mean_squared_log_error
            }
 
# =============================================================================
# Create a dictionary to store all the data sets created.       
# =============================================================================
    
        master_dict = {}
        
        master_dict[params_dict['ID']] = {}
        
        

# =============================================================================
# Create the volatility equation model
# ============================================================================= 

        if params_dict['vol'] == "GARCH" or params_dict['vol'] == "EGARCH":
            
            vol_model = vol_dict[params_dict['vol']](
                p = int(params_dict['p']),
                q = int(params_dict['q']),
                o = int(params_dict['o'])
                )
            
        elif params_dict['vol'] == 'ARCH':

            vol_model = vol_dict[params_dict['vol']](
                p = int(params_dict['p']),
                )

            
        elif params_dict['vol'] == 'FIGARCH':
            
            vol_model = vol_dict[params_dict['vol']](
                p = int(params_dict['p']),
                q = int(params_dict['q'])
             
                )
# =============================================================================
# Iterate over the cross-validation generator to obtain the dates related to
# the observations used in the model. Create the model based on the arguments
# passed.
# =============================================================================
        i = 0
        

        
        for train_dates, val_dates in cv_gen:
     
            model = mean_dict[params_dict['mean']](
                df_return.loc[train_dates],
                volatility = vol_model,
                distribution = dist_dict[params_dict['dist']]
                )
            
            
# =============================================================================
# Now fit the model, generate forecasts, compute the values of the loss func-
# tions, save the parameters of each iteration and the model specifications
# in a separate data frame. Aggregate all dataframes in a dictionary and return
# it.
# =============================================================================
            warnings.warn = lambda *a, **kw: False
            convergence_successful = False
            fitted_model = model.fit(disp = False, options = {'maxiter': 10000})
            
            
            if fitted_model.optimization_result['success'] == True:
                try:
                    if i == 0:
                        
                        i += 1
                        
                        df_forecasts = fitted_model.forecast(
                            horizon = h,
                            method = forecast_method
                            ).variance
                        
                        df_mparams = fitted_model.params.to_frame()
                        
                        df_var_proxy_transposed = df_var.loc[val_dates].T
                        
                 
                        multi_index_columns = pd.MultiIndex.from_product([
                            df_var.columns.tolist(),
                            df_forecasts.columns.tolist()
                            ])
                        
                        df_var_validation = pd.DataFrame(
                            data = [df_var_proxy_transposed.values.flatten()],
                            index = df_forecasts.index, 
                            columns = multi_index_columns
                            )
                        
                        
                    else:
                        
                        i += 1
                        
                        df_forecasts_temp = fitted_model.forecast(
                            horizon = h,
                            method = forecast_method
                            ).variance
                        
                        df_forecasts = pd.concat(
                            [df_forecasts, df_forecasts_temp], axis = 0
                            )
                        
                        
                        df_mparams = pd.concat(
                            [df_mparams, fitted_model.params.to_frame()], axis = 1
                            )
                        
                        df_var_proxy_transposed = df_var.loc[val_dates].T
                        
                        df_var_validation_temp = pd.DataFrame(
                            data = [df_var_proxy_transposed.values.flatten()],
                            index = df_forecasts_temp.index,
                            columns = multi_index_columns
                            )
                        
                        df_var_validation = pd.concat(
                            [df_var_validation, df_var_validation_temp], axis = 0
                            )
                        
                        convergence_successful = True
                
                except:
                    
                    convergence_successful = False
                    break                       
            else:
                
                convergence_successful = False
                   
                break

 
 
# =============================================================================
# Evaluation of the model according to the loss functions passed in the list
# as argument to the function.

# Start an evaluation data frame with the values of -100. Because non of the
# evaluation measures can be negative, this will hint that there has been
# a convergence problem in the computation of the cross validation.
# =============================================================================
        try:
            df_eval = pd.DataFrame(
                data = -100.00,
                columns = df_var_validation.columns,
                index = loss_funcs
                )
# =============================================================================
# Create an exception for the case that the convergence problem arises in the
# first iteration.
# =============================================================================
        except: # +++ exception captures the case the convergence problem happens
                # +++ in the first run
            df_var_proxy_transposed = df_var.iloc[:h].T
            
            multi_index_columns = pd.MultiIndex.from_product([
                df_var_proxy_transposed.columns.tolist(),
                ['h.{}'.format(int(i) for i in range(1, h + 1))]
                ])
                
  
            
            df_eval = pd.DataFrame(
                data = -100.00,
                index = [pd.to_datetime('2000-01-01')],
                columns = multi_index_columns
                )

        if convergence_successful:
            
            for column in df_var_validation:
                
                for loss_func in loss_funcs:
                    
                    try:
                        df_eval.loc[loss_func, column] = loss_functions_dict[loss_func](
                            df_var_validation[column], df_forecasts[column[1]]
                            )
                    except:
                        df_eval.loc[loss_func, column] = -500
                            
# =============================================================================
# The columns of df_mparams are named params, params ... the data set will be 
# transposed, such that the param-names are in the columns and the rows will be
# reindexed to 1...T, where T is the total number of iterations.

# Next, the dataframes created thus far are saved to the master dictionary and
# returned. In the case of non-convergence of the parameters, the dictionary
# is filled with 'Convergence Error'.

# We will rename the axis 0 (index-col) of all dataframes with their respective
# ids and notice that we will also create an evaluation set
# =============================================================================
            
            df_mparams = df_mparams.T
            df_mparams.index = range(1, df_mparams.shape[0] + 1)
            df_mparams.rename_axis(
                'ID_{}'.format(params_dict['ID']), axis = 0, inplace = True
                )    
            master_dict[params_dict['ID']]['model_params'] = df_mparams
            
            
            df_var_validation.rename_axis(
                'ID_{}'.format(params_dict['ID']), axis = 0, inplace = True
                )
            master_dict[params_dict['ID']]['val_var'] = df_var_validation
            
            df_forecasts.rename_axis(
                'ID_{}'.format(params_dict['ID']), axis = 0, inplace = True
                )
            master_dict[params_dict['ID']]['val_forecast_var'] = df_forecasts
            
# =============================================================================
# We also create a dataframe to store the standardized residuals and volatili-
# ties, which can be used in a later step to assess the model. Notice that this
# is only meaningful for the cross-validation method from origin.
# =============================================================================
            if cv_type == 'origin':
                df_resid = fitted_model.resid.rename_axis(
                    'ID_{}'.format(params_dict['ID']), axis = 0
                    )
                
                
                df_stdresid = fitted_model.std_resid.rename_axis(
                    'ID_{}'.format(params_dict['ID']), axis = 0
                    )
                
                
                
                df_condvol = fitted_model.conditional_volatility.rename_axis(
                    'ID_{}'.format(params_dict['ID']), axis = 0
                    )
                
                df_joined = pd.concat([df_resid, df_stdresid, df_condvol], axis = 1)
                master_dict[params_dict['ID']]['resid_and_cvol'] = df_joined
                
                df_params_cov = fitted_model.param_cov.rename_axis(
                    'ID_{}'.format(params_dict['ID']), axis = 0
                    )
                master_dict[params_dict['ID']]['params_cov'] = df_params_cov
                
                df_params_stats = pd.concat(
                    [fitted_model.params,
                     fitted_model.std_err,
                     fitted_model.tvalues,
                     fitted_model.conf_int(),
                     fitted_model.pvalues
                     ], axis = 1
                    )
                
                df_params_stats = df_params_stats.rename_axis(
                    'ID_{}'.format(params_dict['ID']), axis = 0
                    )
                master_dict[params_dict['ID']]['params_stats'] = df_params_stats
        else:
            
            master_dict[params_dict['ID']]['val_var'] = 'Convergence Error'
            master_dict[params_dict['ID']]['val_forecast_var'] = 'Convergence Error'
            master_dict[params_dict['ID']]['model_params'] = 'Convergence Error'
            
            if cv_type == 'origin':
                master_dict[params_dict['ID']]['resid_and_cvol'] = 'Convergence Error'
                master_dict[params_dict['ID']]['params_stats'] = 'Convergence Error'
                master_dict[params_dict['ID']]['params_cov'] = 'Convergence Error'
        
        
        df_eval.rename_axis(
            'ID_{}'.format(params_dict['ID']), axis = 0, inplace = True
            )
        
        master_dict[params_dict['ID']]['eval'] = df_eval
        
        
        return master_dict

    def Cross_validation_evaluation(self,
                                    df_var,
                                    cv_type:str = 'origin',
                                    h:int = 5,
                                    size: int = 250,
                                    seed:int = None,
                                    max_workers:int = os.cpu_count(),
                                    loss_funcs:List[str] = ['mse', 'r2'],
                                    mean: Union[List[str], None] = None,
                                    vol: Union[List[str], None] = None,
                                    pqo: Union[List[str], None] = None,
                                    pqo_ismax: bool = True,
                                    dist: Union[List[str], None] = None,
                                    to_db: bool = False,
                                    engine = None
                                    ):
        warnings.filterwarnings('ignore', category = OptimizeWarning)
# =============================================================================
# Start by checking whether the df_var has pd.DatatimeIndex instance and if all
# the values in the index column is also in the index column of the dataframe
# with the returns, passed at the time of instantiation (see __init__).
# =============================================================================
        assert isinstance(df_var.index, pd.DatetimeIndex), " ".join([
            'df_var index column must be in pd.DatetimeIndex format.'
            ])
        
        assert self.df.index.isin(df_var.index).all(), " ".join(
            ['One or more dates contained in the data-set holding the returns',
             'is not present in the data-set containing the variances.']
            )
# =============================================================================
# Assert that if the data is to be stored in a database, that the engine is
# passed.
# =============================================================================
        if to_db:
            assert engine is not None, " ".join([
                'An sql-alchemy engine must be passed to stored the data to'
                'the data-base.'
                ])
# =============================================================================
# Check if arguments passed to the function are valid and by
# formatting the strings transforming them to lower, upper or proper cases.
# =============================================================================
        if cv_type:
            
            self.__assert_instance(
                args = cv_type, inst = str, arg_name = 'cv_type'
                )
            
            cv_type = cv_type.lower()
            
            self.__check_if_args_are_valid(dict_args = {'cv_type': cv_type})
            
            
        if loss_funcs:
            
            self.__assert_instance(
                args = loss_funcs, inst = list, arg_name = 'loss_funcs'
                )
            
            
            
        if mean is not None:
            
            self.__assert_instance(args = mean, arg_name = 'mean', inst = list)
            
            mean = self.__format_strings(
                list_strings = mean, format_type = 'title'
                )
            
            self.__check_if_args_are_valid(
                dict_args = dict({'mean': mean})
                )
        else:
            
            mean = self.mean
            
        if vol is not None:
            
            self.__assert_instance(args = vol, arg_name = 'vol', inst = list)
            
            vol = self.__format_strings(
                list_strings = vol, format_type = 'upper'
                )
            
            self.__check_if_args_are_valid(
                dict_args = dict({'vol': vol})
                )
        
        else:
            
            vol = self.vol
            
        if pqo is not None:
    
            self.__assert_instance(args = pqo, arg_name = "pqo", inst = tuple)
            self.__check_if_args_are_valid(
                dict_args = dict({'pqo': list(pqo)})
                )
        else:
            
            pqo = (self.max_comb, self.max_comb, self.max_comb)
            
        if dist is not None:
            
            self.__assert_instance(args = dist, arg_name = 'dist', inst = list)
            
            dist = self.__format_strings(
                list_strings = dist, format_type = 'lower'
                )
            
            self.__check_if_args_are_valid(
                dict_args = dict({'dist': dist})
                )
# =============================================================================
# Query the df_args to obtain specifications that meet criteria and
# filter the dataset to obtain those rows.
# =============================================================================

        queried_indices = self.__query_specification(
            mean_list = mean,
            vol_list = vol,
            pqo_tuple = pqo,
            dist_list = dist,
            pqo_ismax = pqo_ismax
            )
        
        df_filtered = self.df_args.loc[queried_indices]
        
        df_filtered.sort_values(by = ['vol', 'mean', 'dist', 'p', 'q', 'o'],
                                inplace = True)
        
# =============================================================================
# reindex the dataframe so that the index column has ordered indices
# =============================================================================
        df_filtered.index = range(1, df_filtered.shape[0] + 1)
        
        df_filtered.rename_axis('ID', inplace = True)
        df_filtered.reset_index(inplace = True)
        
        processes_input = df_filtered.to_dict(orient = 'records')
        
        df_return = self.df

        partial_CV_score = partial(
            GarchModels.CV_score,
            df_return,
            df_var,
            loss_funcs,
            h,
            size,
            seed,
            cv_type
            )
        
# =============================================================================
# Use ProcessPoolExecutor to parallelize the process
# =============================================================================

        
        
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = list(
                    tqdm(
                        executor.map(partial_CV_score, processes_input),
                        ncols = 100,
                        desc = 'Cross-Validating Models',
                        total = len(processes_input)
                        )
                    )
        finally:
            executor.shutdown(wait=True)
            
# =============================================================================
# join all the dictionaries in another dictionary with two keys, one containing
# the df_filtered as value (with the model ids and specifications) and one
# containing the master dictionaries
# =============================================================================


        
        result_dict = {}
        result_dict['spec'] = df_filtered
        result_dict['models'] = {}
        for dictionary in results:
            
            for key, value  in dictionary.items():
            
                result_dict['models']['ID_{}'.format(key)] = value
            
        
        
        print('{}\nModel estimation completed.'.format('='*50))

# =============================================================================
# add results to database or return the dictionary with the results.
# =============================================================================
        if to_db:

            self.dict_to_database(
                result_dict = result_dict,
                connection = engine,
                if_exists = 'fail'
                )
            
            self.validation_var_to_database(
                result_dict = result_dict,
                connection = engine,
                if_exists = 'fail'
                )
            
            self.aggregate_lossfuncs(
                result_dict = result_dict,
                connection = engine,
                if_exists = 'fail'
                )
            
            self.aggregate_residuals_and_condvol(
                result_dict = result_dict,
                connection = engine,
                if_exists = 'fail'
                )

            self.aggregate_forecasts(
                result_dict = result_dict,
                connection = engine,
                if_exists = 'fail'
                )
            
        else:
            
            return result_dict
        
            
# =============================================================================
# Create functions to aggregate and save the results yielded by the cross-vali-
# dations.    
# =============================================================================

    def dict_to_database(self,
                         result_dict:dict,
                         connection:Engine,
                         if_exists:str = 'fail'):
        """
        Function to store all pandas dataframes in the result_dict created
        by the Cross_validation_evaluation function to a locally stored data-
        base passed as argument to the connection. The table with the IDs
        of those models whose convergence failed is saved with the name "Con-
        vergence Error", all other tables are saved with the pattern "ID_i_key",
        where i is an integer related to the model ID and the key is the name of
        the key in the result_dict that has the given table as value.

        Parameters
        ----------
        result_dict : dict
            The dictionary returned by Cross_validation_evaluation
        connection : Engine
            A connection to the database. If the argument is None, only the 
            pandas dataframe with the aggregated loss-values will be returned.
            The default is None.
        if_exists : str, optional
            Controls the behaviour of the engine in the case a table with the
            same name exists Valid arguments are:
                - fail
                - replace
            The default is 'fail'.

        Returns
        -------
        None.

        """
        
        ids_convergence_error = []
        for model_id, model_data in tqdm(
                result_dict['models'].items(),
                desc = 'Saving results to database',
                ncols = 100,
                unit = 'it'
                ):
            for key, dataframe in model_data.items():
                
                if isinstance(dataframe, pd.DataFrame):
                    table_name = '{}_{}'.format(model_id, key)
                    
                    if key in ['val_var', 'val_forecast_var', 'eval']:
                        try:
# =============================================================================
# create expection to handle the case where the index has been reseted already
# =============================================================================
                            dataframe.reset_index(inplace = True)
                            export_index = False
                        except:
                            export_index = False
                    else:
                        export_index = True
                        
                    dataframe.to_sql(
                        con = connection,
                        name = table_name,
                        if_exists = if_exists,
                        index = export_index
                        )

# =============================================================================
# Notice that if the instance is not a data frame, then it will be a string
# containing the value "Convergence Error". For those ids a new data frame will
# be created, giving the user the possibility to assess which model failed to
# converge.
# =============================================================================
                else:
                    ids_convergence_error.append(model_id)
                    
        
        if len(ids_convergence_error) > 0:
            
            ids_convergence_error = list(set(ids_convergence_error))
            df_convergence_error = pd.DataFrame(
                data = ids_convergence_error,
                columns = ['ID']
                )
            
            df_convergence_error.drop_duplicates(inplace = True)
            
            df_convergence_error.to_sql(
                con = connection,
                name = 'Convergence Error',
                if_exists = if_exists,
                index = False
                )
        
# =============================================================================
# Also save the model specifications to the database
# =============================================================================
        
        result_dict['spec'].to_sql(
            con = connection,
            name = 'Specification Table',
            if_exists = if_exists,
            index = False
            )
        print('{}\nDictionaries to data-base completed.'.format('='*50))
    
            
    def aggregate_lossfuncs(self,
                            result_dict:dict,
                            connection:Engine = None,
                            if_exists:str = 'fail'):
        """
        Function to aggregate the loss-function values computed by the function
        Cross_validation_evaluation, and to save them to a database in form
        of a unified pandas dataframe. The table is saved with the name 
        "Agg. Loss Function Values".
        
        The table will be saved with the name 'Agg. Loss Function Values'
        Parameters
        ----------
        result_dict : dict
            The dictionary returned by the funciton Cross_validation_evaluation
        connection : Engine, optional
            A connection to the database. If the argument is None, only the 
            pandas dataframe with the aggregated loss-values will be returned.
            The default is None.
        if_exists : TYPE, optional
            Controls the behaviour of the engine in the case a table with the
            same name exists Valid arguments are:
                - fail
                - replace
            The default is 'fail'.

        Returns
        -------
        An aggregated pandas dataframe containing the losses for all models
        passed as argument to the function.

        """
        
        counter = 0
        for model_id in tqdm(
            result_dict['models'].keys(),
            desc = 'Aggregating Losses',
            unit = 'it',
            ncols = 100
            ):
                
        
            if isinstance(result_dict['models'][model_id]['eval'], pd.DataFrame):
                if counter == 0:
                    
                    df_losses = (
                        result_dict['models'][model_id]['eval'].copy(deep = True)          
                        )
                    
                    df_losses['ID'] = model_id.split('_')[1]
                    
                    counter += 1
                else:
                    
                    df_losses_temp = (
                        result_dict['models'][model_id]['eval'].copy(deep = True)
                        )
                    
                    df_losses_temp['ID'] = model_id.split('_')[1]
                    
                    df_losses = pd.concat([df_losses, df_losses_temp], axis = 0)
                    
        if connection is not None:
            df_losses.rename_axis('metric', inplace = True)
            df_losses.reset_index(inplace = True)
            df_losses.to_sql(
                name = 'Agg. Loss Function Values',
                con = connection,
                index = False,
                if_exists = if_exists
                )
            
        print('{}\nAgg. Loss Function Values completed'.format('='*50))
    
    
    def aggregate_residuals_and_condvol(self,
                                        result_dict:dict,
                                        connection:Engine = None,
                                        if_exists:str = 'fail'):
        """
        Function to store the residuals, standardized residuals and conditio-
        nal volatility in a single dataframe in long-format. The table is saved
        with the name 'Agg. Residuals and Cond. Volatility'

        Parameters
        ----------
        result_dict : dict
            The dictionary returned by the funciton Cross_validation_evaluation
        connection : Engine, optional
            A connection to the database. If the argument is None, only the 
            pandas dataframe with the aggregated residua and conditional vola-
            tilities will be returned.
            The default is None.
        if_exists : TYPE, optional
            Controls the behaviour of the engine in the case a table with the
            same name exists Valid arguments are:
                - fail
                - replace
            The default is 'fail'.

        Returns
        -------
        A pandas dataframe in long format containing a column for the resida,
        the standardized residua, the conditional variance, and the model id.

        """
        
        counter = 0 
        for model_id in tqdm(
            result_dict['models'].keys(),
            desc = 'Aggregating Residuals and cond. Vol.',
            unit = 'it',
            ncols = 100
            ):
                

            
            if isinstance(result_dict['models'][model_id]['resid_and_cvol'], pd.DataFrame):
                if counter == 0:
                    
                  
                    df_resid_codval = (
                        result_dict['models'][model_id]['resid_and_cvol'].copy(deep = True)
                        )
                    
                    df_resid_codval['ID'] = model_id.split('_')[1]
                    
                    counter += 1
                    
                else:
                    
                    df_resid_codval_temp = (
                        result_dict['models'][model_id]['resid_and_cvol'].copy(deep = True)
                        )
                    
                    df_resid_codval_temp['ID'] = model_id.split('_')[1]
                    
                    df_resid_codval = pd.concat([
                        df_resid_codval, df_resid_codval_temp
                        ], axis = 0)
                
# =============================================================================
# If a connection was passed as argument, the dataframe will be stored in the
# database. Otherwise the dataframe will be returned without saving.
# =============================================================================
        if connection is not None:
            df_resid_codval.to_sql(
                name = 'Agg. Residuals and Cond. Volatility',
                con = connection,
                index = True,
                if_exists = if_exists
                )
            
        print('{}\nAgg. Residuals and Cond. Volatility completed'.format('='*50))
    
    def aggregate_forecasts(self,
                            result_dict:dict,
                            connection:Engine = None,
                            if_exists:str = 'fail'):
        """
        Function to aggregate all dataframes containing the variance forecasts
        stored in the result_dict['val_forecast_var'] in a single dataframe in
        long format.

        Parameters
        ----------
        result_dict : dict
            The dictionary returned by the funciton Cross_validation_evaluation
        connection : Engine, optional
            A connection to the database. If the argument is None, only the 
            pandas dataframe with the aggregated forecasts will be returned.
            The default is None.
        if_exists : TYPE, optional
            Controls the behaviour of the engine in the case a table with the
            same name exists Valid arguments are:
                - fail
                - replace
            The default is 'fail'.

        Returns
        -------
        A pandas dataframe in long format containing the h-steps ahead forecasts
        one in each column; a column with the dates indicating the boundaries
        of observations used to produce the h-step ahead forecasts, and a column
        containing the model ID to uniquely identifying the models.
        """
        
        counter = 0
        for model_id in tqdm(
            result_dict['models'].keys(),
            desc = 'Aggregating Forecasts datasets',
            unit = 'it',
            ncols = 100
            ):
                
         
            if isinstance(
                    result_dict['models'][model_id]['val_forecast_var'], 
                    pd.DataFrame
                    ):
                
                if counter == 0:
                    
                    df_forecasts = (
                        result_dict['models'][model_id]['val_forecast_var'].copy(deep = True)
                        )
                    
                    df_forecasts['ID'] = model_id.split('_')[1]
                 
                    counter += 1
                    
                else:
                    
                    df_forecasts_temp = (
                        result_dict['models'][model_id]['val_forecast_var'].copy(deep = True)
                        )
                    
                    df_forecasts_temp['ID'] = model_id.split('_')[1]
                    
                    df_forecasts = pd.concat(
                        [df_forecasts, df_forecasts_temp], axis = 0
                        )
       
        if connection is not None:
            df_forecasts.rename_axis('Trading Date', inplace = True)
            df_forecasts.reset_index(inplace = True)
            df_forecasts.to_sql(
                name = 'Agg. Forecasted Variances',
                con = connection,
                index = False,
                if_exists = if_exists
                )
            
        print('{}\nAgg. Forecasted Variances to data-base completed.'.format('='*50))
    
    def validation_var_to_database(self,
                            result_dict:dict,
                            connection:Engine = None,
                            if_exists:str = 'fail'):
        """
        Function to save the validation data set containing the target-varian-
        ces used in the cross-validation.
        
        The table will be saved with the name 'Agg. Loss Function Values'
        Parameters
        ----------
        result_dict : dict
            The dictionary returned by the funciton Cross_validation_evaluation
        connection : Engine, optional
            A connection to the database. If the argument is None, only the 
            pandas dataframe with the aggregated loss-values will be returned.
            The default is None.
        if_exists : TYPE, optional
            Controls the behaviour of the engine in the case a table with the
            same name exists Valid arguments are:
                - fail
                - replace
            The default is 'fail'.

        Returns
        -------
        A dataframe containing the variance proxies used to compute the loss
        functions in the cross-validation method.

        """
        
# =============================================================================
# Notice that the dataset is common to all models, hence it is enough to only
# save the dataset once.
# =============================================================================
        is_dataframe = False
        i = 0
        while not is_dataframe:
            i += 1
            df_valvar = result_dict['models']['ID_{}'.format(i)]['val_var']
            if isinstance(df_valvar, pd.DataFrame):
                try:
                    df_valvar.rename_axis('Trading Date', inplace = True)
                    df_valvar.reset_index(inplace = True)
                    is_dataframe = True
                except:
# =============================================================================
# creates exception for the case a column named 'trading date already exists'
# =============================================================================
                    is_dataframe = True
        
        df_valvar.to_sql(
            name = 'Validation Variance',
            con = connection,
            index = False,
            if_exists = if_exists
            )
        
        print('{}\nValidation variance to data-base completed.'.format('='*50))
    

#%% Section 2: Definition of Out-of-Sample Forecaster-Class
#%%

# =============================================================================
# Create function to generate out-of-sample forecasts
# =============================================================================
class garchForecaster():
    
    def __init__(self, h, seed = None):
        self.seed = seed
        self.h = h
        
    def generate_forecast(self, model_obj, df, df_train, seed = None):
    
        seed = self.seed if seed is None else seed
        


        
# =============================================================================
# Start by obtaining the test set (recall that we are including the last-5 observations used as validation in the train set)
# =============================================================================
        df_test = df[df.index > df_train.index[-1]].copy()
# =============================================================================
# Obtain the model components using regular expression
# =============================================================================
        pattern = r'(.*Mean)\(.*volatility:\s(.*)\(.*distribution:\s(.*)\)'
        specification_str = str(model_obj.model)
        match = re.match(pattern, specification_str, re.DOTALL)

# =============================================================================
#  Obtain the orders of the process
# =============================================================================
        order_pattern = r'(p|q|o): (\d+)'
        order_match = re.findall(order_pattern, specification_str)

# =============================================================================
# create a dictionary to serve as argument to the volatility specification
# =============================================================================
        order_dict = {
            key: int(value) for key, value in order_match
        }
# =============================================================================
# create a dictionary with the p q o values
# =============================================================================
        mean_dict = {
            'Constant Mean': ConstantMean,
            'Zero Mean': ZeroMean
        }
        
        vol_dict = {
            'EGARCH': EGARCH,
            'GARCH': GARCH,
            'ARCH': ARCH,
            'FIGARCH': FIGARCH
        }
        
        dist_dict = {
            'Normal distribution': Normal,
            'Generalized Error Distribution distribution': GeneralizedError,
            "Standardized Student's t distribution": StudentsT,
            "Standardized Skew Student's t distribution": SkewStudent
        }

# =============================================================================
#  Create a set of random seed to use in the simulation. This seed is dependent on the seed passed as argument.
#   We create as many random seed as we create forecasts 
# =============================================================================
        np.random.seed(seed)
        random_vector = np.random.randint(1, df_test.shape[0] * 10, size = (df_test.shape[0], 1)).flatten()

# =============================================================================
#  itarte over the df_test set and increment the df_train set by one observation. This is needed, as the arch-package
# always use the last observation to forecast the next one
# =============================================================================
        df_forecasts = pd.DataFrame()
        for i in range(df_test.shape[0]):
    
            df_data = pd.concat([df_train, df_test[:i+1]]).copy()

# =============================================================================
# Build the model based on the model parameters passed in the model_obj and pass a random seed to the distribu-
# tion to ensure that the results are reproducible.
# =============================================================================
        
            model_fixed = mean_dict[match.group(1)](
                df_data, 
                volatility = vol_dict[match.group(2)](**order_dict),
                distribution = dist_dict[match.group(3)](seed = random_vector[i])
            )
# =============================================================================
# create the model using the fixed parameters (i.e., fit the model)
# =============================================================================
            model_fitted_fixed = model_fixed.fix(params = model_obj.params)
           
# =============================================================================
# use the mode to generate forecast
# =============================================================================
            forecasted_values = model_fitted_fixed.forecast(horizon = int(self.h), method = 'simulation')
            if df_forecasts.empty:
                df_forecasts = forecasted_values.variance
            else:
                df_forecasts = pd.concat([df_forecasts, forecasted_values.variance], axis = 0)
        return df_forecasts

    def generate_target(self, df_var):

        h = self.h
# =============================================================================
#  Create a dictionary to store the different proxies of variance.
# =============================================================================
        dict_var = {}

# =============================================================================
# iterate over the columns of the df_var (each column will be a variance proxy and store them in the dict_var)
# =============================================================================
        df_var = df_var.copy() if isinstance(df_var, pd.DataFrame) else df_var.to_frame().copy()
        for column in df_var.columns.tolist():
            dict_var[column] = df_var[column].copy().to_frame()

# =============================================================================
# The variance dataset (target) will be formated in the same fashion as the forecast data set. The arch-package
# produces (as default value) the data frame so, that its index are the dates, and the columns are the 1- ... h-step
# ahead forecasts made at that date.
# We can mimic this by creating new columns in the dataframes dict_var where each one of the columns in shifted
# by -h, h = 1, \dots, H.
# =============================================================================
        for df in dict_var.values():
            for i in range(1, h + 1):

                df['h.{}'.format(i)] = df[df.columns[0]].shift(-i)
# =============================================================================
#  Drop the nan-values from the dataset
# =============================================================================
            df.dropna(inplace = True)
        return dict_var

    def evaluate_performance(self, df_true, df_forecast, model_name = None, compute_cumulative_mse = True):

# =============================================================================
#  Start by leaving only the indices that are common to both datasets. The df_true has higher priority, as without
#  the real value a forecast cannot be evaluated.
# =============================================================================
        h = self.h
        common_indices = df_true.index.intersection(df_forecast.index)
# =============================================================================
# Make sure that the columns are properly set
# =============================================================================
        columns = ['h.{}'.format(i) for i in range(1, h + 1)]
        df_true = df_true.loc[common_indices, columns].copy()
        df_forecast = df_forecast.loc[common_indices, columns].copy()


# =============================================================================
# Compute the squared forecast error
# =============================================================================
        df_sqforecast_error = (df_true - df_forecast) ** 2

# =============================================================================
# create 5-day average and df_MSE
# =============================================================================
        df_sqforecast_error['avg'] = df_sqforecast_error.mean(axis = 1)
        df_MSE = df_sqforecast_error.mean(axis = 0)
        df_MSE = df_MSE.to_frame().T
# =============================================================================
# Join the data set df_true and df_forecast in a single one
# =============================================================================
        df_true['type'] = 'target'
        df_forecast['type'] = 'forecast'
        df_used_data = pd.concat([df_true, df_forecast])

# =============================================================================
# If a model name is passed, we create a column in each one of the dataframes to include the model name
# =============================================================================
        if model_name is not None:
            df_used_data['model_name'] = model_name
            df_sqforecast_error['model_name'] = model_name
            df_MSE['model_name'] = model_name
        
        if compute_cumulative_mse:
            df_sqforecast_error['cumulative_avg_mse'] = df_sqforecast_error['avg'].expanding().mean()
# =============================================================================
# return all the dataframes used in the evaluation in a dictionary
# =============================================================================
        dict_result = {
            'data': df_used_data,
            'squared_error': df_sqforecast_error,
            'mse': df_MSE
        }
        return dict_result

    def __enter__(self):
        print('Starting Test-Set Forecast and Evaluation')
        print('-'*80)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print('-'*80)
        print('Evaluation completed.')
        

