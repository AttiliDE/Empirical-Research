#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 02:03:07 2024

@author: ferreira
"""

#%% Section 0: Imports
#%%
import pandas as pd
import numpy as np
import rpy2
import rpy2.robjects as ro
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri, numpy2ri
import os
from sqlalchemy import create_engine
from tqdm import tqdm
from datetime import datetime
import scipy.stats as stats

#%%

class aSPA():
    
    def __init__(self, 
                 auto_length_selection: bool = True, 
                 selection_params: dict = None,
                 seed = None):
        
        self.auto_length_selection = auto_length_selection
        self.seed = seed
# =============================================================================
# Assert that the parameters passed in the selection params are valid
# =============================================================================
        valid_params = ['cores', 'k', 'verbose', 'seed', 'sub_sample', 'n_iter']
        
        if selection_params is not None:
            
            for param in selection_params:
                
                assert param in valid_params, " ".join([
                    f'{param} is not a valid argument for the automated',
                    f'length selection. Valid params are: {valid_params}'
                    ])
            self.selection_params = selection_params
        else:
            self.selection_params = {
                'cores': os.cpu_count() - 1,
                'k': 'one-sided',
                'verbose': False,
                'seed': self.seed,
                'sub_sample': 100,
                'n_iter': 1000
                }
    
    def bulk_aSPA(self, 
                  j:str, 
                  LossDiff_dict: dict, 
                  weights: list, 
                  L:int = None, 
                  B:int = 1000,
                  alpha = 0.05,
                  bootstrap = True):
        """
        Function to compute aSPA statistics for all Loss Differentials passed
        in the dictionary LossDiff_dict

        Parameters
        ----------
        j : str
            The name of the model with prior knowledge. The null hypothesis
            states that Li - Lj \leq 0, i.e. that model j has lower or equal
            predictive ability than model i.
        LossDiff_dict : dict
            A dictionary with the model names as keys and a pandas dataframe
            with the loss differential as values.
        weights : list
            The weight to give for each of the h-step ahead forecast to compute
            the weighted loss differential.
        L : int, optional
            The length of the window to use in the MBB. 
            If auto-length selection is passed as argument to the class
            this value is ignored.
            Default: None.
        B : TYPE, optional
            The number of samples to use in the bootstrap.
            Default: 1000
        alpha: float, optional
            The significance level for the test-statistic (a value between 0-1).
            Default: 0
        Returns
        -------
        A pandas dataframe containing the p-value of the boostraped t-statistic,
        the aSPA statistis (formula 10 of the original paper). The model i 
        (passed as key in the dictionary) and model j names, a boolean column
        if the H0 is to be rejected, as well as the number of bootstraps samples
        and length window used in the bootstrap.

        """
        start = datetime.now()
        formatted_time = start.strftime("%H:%M:%S")
        print('Starting estimation. Time {}.'.format(formatted_time))
        print('-'*80)
        print('\n')
        df_bootstraped_dist = pd.DataFrame()
        results = {}
        if bootstrap == True:
            for model, df in tqdm(LossDiff_dict.items(), 
                                  desc = 'Computing aSPA',
                                  unit = 'it',
                                  ncols = 100):
                results_temp = self.Test_aSPA(LossDiff = df, weights = weights, L = L, B = B)
                results[model] = pd.DataFrame(
                    data = results_temp[0],
                    columns = ['p-value', 't_aSPA', 'crit-val', 'B', 'L']
                    )
    
                results[model]['model_j'] = j
                results[model]['model_i'] = model
# =============================================================================
# Save the boostrapped distribution to df_boostraped_dist    
# =============================================================================
                if df_bootstraped_dist.empty:
                    
                    df_bootstraped_dist = pd.DataFrame(
                        data = results_temp[1],
                        columns = ['t_aSPA_b']
                        )
                    
                    df_bootstraped_dist['model_j'] = j
                    df_bootstraped_dist['model_i'] = model
                
                else:
                    
                    df_temp = pd.DataFrame(
                        data = results_temp[1],
                        columns = ['t_aSPA_b']
                        )
                    df_temp['model_j'] = j
                    df_temp['model_i'] = model
                    
                    df_bootstraped_dist = pd.concat([df_bootstraped_dist, df_temp], axis = 0)
# =============================================================================
# join all the dataframes
# =============================================================================
            result = pd.concat(results.values())
            result['Reject H0'] = result['p-value'].apply(lambda x: True if x < alpha else False)
            result.reset_index(drop = True, inplace = True)
            end = datetime.now()
            formatted_time = end.strftime("%H:%M:%S")
            print('\nEstimation completed. Time {}.\n'.format(formatted_time))
            print('-'*80)
            end = datetime.now()
            time_elapsed = end - start
            print('Time elapsed in h. {}'.format(round((time_elapsed.seconds / 3600), 3)))
            print('\n')
            return result, df_bootstraped_dist

# **************************************************       
        else:
            
            for model, df in tqdm(LossDiff_dict.items(), 
                                  desc = 'Computing aSPA',
                                  unit = 'it',
                                  ncols = 100):
            
# =============================================================================
# If bootstrap did not equal true we will compute the standard implementation
# of the test. I.e. we will rely on the large-sample properties 
# =============================================================================
                if self.seed is not None:
                    np.random.seed(self.seed)
                    
                weights = np.array(weights)
                weighted_LossDiff = df @ weights
# =============================================================================
# obtain number of daily loss differential and calculate the weighted loss-diff
# =============================================================================

                TT = weighted_LossDiff.shape[0]
            
# =============================================================================
# Compute the sample mean of the loss differential
# =============================================================================
                #[2]
                d_ij = weighted_LossDiff.mean()
        
# =============================================================================
# Compute the test statistic (Formula 10 of the article)
# =============================================================================
                #[3]
                numerator = np.sqrt(TT) * d_ij
                variance = self.QS(weighted_LossDiff)
                denominator = np.sqrt(variance)
# =============================================================================
# Create data frame to hold the test statistic, the variance 
# =============================================================================
            
            
                t_aSPA = (numerator / denominator)[0][0]

                critical_val = stats.norm.pdf(1 - alpha)
                p_value = round(1 - stats.norm.cdf(t_aSPA), 4)
        
                reject_null = True if p_value < alpha else False
                
# =============================================================================
# Create a dataframe to store the results
# =============================================================================
                df_results = pd.DataFrame({
                    't_aSPA': [t_aSPA],
                    'd_ij': [d_ij],
                    'nobs': [weighted_LossDiff.shape[0]],
                    'svar': [variance[0][0]],
                    'crit-val': [critical_val],
                    'p-value': [p_value],
                    'model_j': [j],
                    'model_i': [model],
                    'reject_h0': [reject_null]
                    })
                
                
                results[model] = df_results
            
            return pd.concat(results.values())

    def estimate_optimal_length(self, series: np.array, params: dict):
        """
        Function to estimate the optimal window length to use in the Moving
        Block Bootstrap. The function implementation is based on an R package.
        Consequently its usage requires the availability of the R language
        and of the packages "parallel" and "blocklength."

        Parameters
        ----------
        series : np.array
            A (1x1) numpy array with the series that will be boostrapped.
        params : dict
            A dictionary of parameters to pass to the R function. Valid parame-
            ters are:
                - cores: the number of cores to use in the cross-validation.
                Default: total cores minus one.
                - sub_sample: the number of samples to use in the cross-valida-
                tion.
                Default: 100
                - k: the type of bootrap, i.e. whether the boostrap is used
                to estimate a distribution, the bias or the variance.
                Default: "one-sided" (distribution)
                - verbose: True if information about the cross-validtion should
                be displayed in the console.
                Default: False
                - seed: The random seed to be passed to the cores in order to
                ensure reproducibility.
                Default: None.

        Returns
        -------
        The optimal window length as a numpy array.

        """

# =============================================================================
# Define r code to calculate optimal window-length for the MBB.
# =============================================================================
        r_code = """
        suppressPackageStartupMessages(library(parallel))
        suppressPackageStartupMessages(library(blocklength))
        suppressPackageStartupMessages(library(quantmod))
        
        compute_optimal_length <- function(series,
                                           cores,
                                           sub_sample = 100,  
                                           k = "one-sided",
                                           verbose = FALSE,
                                           n_iter = 1000,
                                           seed = NULL) {
            
            cluster <- makeCluster(cores)
            on.exit(stopCluster(cluster))
            
            if (!is.null(seed)) {
                    clusterSetRNGStream(cluster, seed)
                    }
            
            result <- hhj(series = as.numeric(series),
                          sub_sample = sub_sample,
                          cl = cluster,
                          k = k,
                          verbose = verbose,
                          n_iter = n_iter)
            
            return(result$`Optimal Block Length`)
            }
        """
        
        with localconverter(ro.default_converter 
                            + pandas2ri.converter 
                            + numpy2ri.converter):
            
            ro.r(r_code)
            hhj_function = ro.globalenv['compute_optimal_length']
            result = hhj_function(series = series, **params)
            
        return result
                
    def Bootstrap_aSPA(self, LossDiff:pd.DataFrame, weights:list, L:int, B:int):
        """
        Function to implement the Moving Block Boostrap of Künsch (1989)

        Parameters
        ----------
        LossDiff : pd.DataFrame
            The Loss differential vector with h-elements (columns)
        weights : list
            The weights to use in the computation of the weighted loss diffe-
            rential.
        L : int
            The size of the block length to use in the bootstrap.
        B : int
            The number of samples used in the bootstrap procedure.

        Returns
        -------
        None.

        """
# =============================================================================
# Set the random seed (if applicable)        
# =============================================================================
        if self.seed is not None:
            np.random.seed(self.seed)
            
        weights = np.array(weights)
# =============================================================================
# obtain number of daily loss differential and calculate the weighted loss-diff
# =============================================================================

        TT = LossDiff.shape[0]
        
        #[1] 
        
        weighted_LossDiff = LossDiff @ weights

# =============================================================================
# Compute the sample mean of the loss differential
# =============================================================================
        #[2]
        d_ij = weighted_LossDiff.mean()
        
# =============================================================================
# Compute the test statistic (Formula 10 of the article)
# =============================================================================
        #[3]
        numerator = np.sqrt(TT) * d_ij
        denominator = np.sqrt(self.QS(weighted_LossDiff))
        
        
        t_aSPA = numerator / denominator
        
# =============================================================================
# Take the mean from the (weighted) loss differential
# =============================================================================
        #[4]
        demeaned_weighted_LossDiff = weighted_LossDiff - d_ij
# =============================================================================
# Create placeholder to save the t_aSPA_b statistics
# =============================================================================
        t_aSPA_b = np.zeros((B, 1))
# =============================================================================
# Create the boostrapped samples
# =============================================================================
        for b in range(B):
            
            #[5]
            sample_ids = self.get_mbb_id(TT = TT, L = L)
            
            b_lossdiff = demeaned_weighted_LossDiff.iloc[sample_ids]
       
            zeta_b = self.MBB_variance(b_lossdiff, L = L)
            
# =============================================================================
# compute test statistic
# =============================================================================
            numerator = np.sqrt(TT) * b_lossdiff.mean()
            t_aSPA_b[b, 0] = numerator / zeta_b
            
        return t_aSPA, t_aSPA_b
    
    def QS_weights(self, x):

# =============================================================================
# Function to compute the weights given an array x for the kernel-function
# =============================================================================
        argQS = 6 * np.pi * x / 5
        w1 = 3 / (argQS ** 2)
        w2 = (np.sin(argQS) / argQS) - np.cos(argQS)
        
        wQS = w1 * w2
        wQS[wQS == 0] = 1
        
        return wQS
    
    def QS(self, y):
        
# =============================================================================
# Obtain the length of the series and the number of h-step ahead predictions
# =============================================================================
        TT = y.shape[0]
        
        
# =============================================================================
# Define the bandwidth for the kernel
# =============================================================================
        bw = 1.3 * TT ** (1/5)
        
# =============================================================================
# Obtain the weights to use in the kernel estimator. Create a sequence of lags
# from 1 to TT-1
# =============================================================================
        lag_seq = np.arange(1, TT) / bw
        weight = self.QS_weights(lag_seq)
        
# =============================================================================
# Notice that different than the original implementation we will only allow for
# the comparison of two series (one loss differential).
# =============================================================================
        
        mean_dev = y - y.mean()
        mean_dev = mean_dev.to_numpy().reshape(-1, 1)
        omega = mean_dev.T @ mean_dev / TT

# =============================================================================
# Compute the HAC variance:        
# =============================================================================
        for j in range(1, TT):
            omega = omega + 2 * weight[j - 1] * (mean_dev[0:TT-j].T @ mean_dev[j: TT]) / TT
   
        return omega
    
    def get_mbb_id(self, TT, L):
        """
        Get_MBB_ID obtains ids of resampled observations using a
        moving block bootstrap with blocks of length L.
    
        Parameters:
        TT (int): The total number of observations in the time series.
        L (int): The block length for the moving block bootstrap.
    
        Returns:
        numpy.ndarray: An array of indices for the resampled observations.
        """
        id = np.zeros(TT, dtype=int)
        id[0] = np.ceil(TT * np.random.rand()).astype(int) - 1  # Python is 0-indexed, so subtract 1
    
        for t in range(1, TT):
            if t % L == 0:
                id[t] = np.ceil(TT * np.random.rand()).astype(int) - 1
            else:
                id[t] = id[t-1] + 1
            
            if id[t] >= TT: 
                id[t] = 0
    
        return id
    
    def MBB_variance(self, y: pd.Series, L:int):
        """
        Function to compute the variance of the Moving Block Boostrapped
        loss differential.

        Parameters
        ----------
        y : pd.Series
            DESCRIPTION.
        L : int
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
# =============================================================================
# Obtain number of observations in the pandas-series
# =============================================================================
        TT = y.shape[0]
        
        y_demeaned = y - y.mean()
        
        K = int(np.floor(TT / L))

        
# =============================================================================
# Now reshape y_demeaned and compute the variance (as in formula 11) of the
# article
# =============================================================================
        temp = y_demeaned[0:int(K * L)].to_numpy().reshape(K, L)
        omega = np.mean(temp.sum(axis = 0) ** 2) / L

        return omega
    
    def Test_aSPA(self, LossDiff:pd.DataFrame, 
                  weights:list, 
                  L:int = None, 
                  B:int = 1000,
                  alpha: float = 0.05):
        """
        Function to comute the aSPA statistic and the p-value of the bootstraped
        t-statistic.

        Parameters
        ----------
        LossDiff : pd.DataFrame
            A pandas dataframe with the Loss Differential containing h-elements
        weights : list
            The weight to give to each one of the  h-step forecast in computing
            the weighted Loss Differential.
        L : int, optional
            The size of the block used in the MBB. If the automated selection
            is chosen, the value passed to this argument is ignored.
            Default: None
        B : int, optional
            The number of samples used in the MBB to compute the test statistic.
            Default: 1000
        alpha: float, optional
            The significance level used in the test statistic (a value between 0-1).
            Default: 0.05
        Returns
        -------
        A dictionary with the p-value, the aSPA test statistic (Note: This is not
        the bootstraped statistic, but formula 10 of the origianl paper.). The
        p-value on the other hand is the \alpha quantile of the bootstrapped
        distribution of the test statistic.

        """


        weights = np.array(weights)
# =============================================================================
# Assert that if the automated selection of the length window is not activated
# that a length (L) argument was passed.
# =============================================================================
        if not self.auto_length_selection:
            
            assert isinstance(L, int), " ".join([
                'An integer values for L (Block size) must be passed to compute',
                'the Moving Block Bootstrap.'
                ])
            
        else:
# =============================================================================
# compute weighted Loss Differential and use the R-function to compute
# the optimal length.
# =============================================================================
            weighted_LossDiff = (LossDiff @ weights).values
        
            
            L = self.estimate_optimal_length(series = weighted_LossDiff,
                                             params = self.selection_params)
            L = int(L[0])
        t_aSPA, t_aSPA_b = self.Bootstrap_aSPA(LossDiff = LossDiff,
                                         weights = weights,
                                         L = L,
                                         B = B)
        
        critical_value = np.quantile(t_aSPA_b, q = 1-alpha)

        p_value = np.mean(t_aSPA < t_aSPA_b)
 

        return  (({
            'p-value': [p_value],
            't_aSPA': [t_aSPA[0, 0]],
            'crit-val': [critical_value],
            'B': [B],
            'L': [L]
            }, t_aSPA_b))
        
    def compute_lossdifferential(self, dict_losses: dict, model_j: str, h:int = 5):
        """
        Function to compute the loss-differentials: dij = Li - Lj.
        
        Parameters
        ----------
        dict_losses : dict
            A dictionary with pandas data-frames containing the losses of the
            models. The key of the dictionary should contain the name of the models
            to uniquely identify the loss differential.
            
            The index of the pandas dataframes must be in date-time format.
            The columns of the dataframes must contain the names h.1, h.2 ..., h.H.
            with H being the max-step ahead forecast. The loss differential is
            computed column-wise. Meaning that column h.1 (h2.,...,h.H) of model j will be 
            subtracted from column h.1 (h.2,...,h.H) of model i. 
        model_j : str
            The name of the model all other models will be compared against.
        h : int, optional
            The total number of step-ahead forecasts in the pandas dataframe.

        Returns
        -------
        A dictionary containing pandas dataframe with the loss differential
        of each of the models against model j.

        """
# =============================================================================
# Iterate over the dictionary and subtract the losses of model j from model i
# =============================================================================
        lossdiff_dict = {}
        for model, df in dict_losses.items():
    
            df_i = df.copy()
            df_j = dict_losses[model_j].copy()
    
# =============================================================================
# Make sure that only common dates are considered
# =============================================================================
            common_index = df_i.index.intersection(df_j.index)
            
            columns = ['h.{}'.format(i) for i in range(1, h + 1)]
            
            df_i = df_i.loc[common_index, columns].copy()
            df_j = df_j.loc[common_index, columns].copy()
# =============================================================================
# Compute loss differential and save to the dictionary
# =============================================================================        
            LossDiff = df_i - df_j
    
            if all(LossDiff.sum() != 0):
                lossdiff_dict[model] = LossDiff
            
        return lossdiff_dict
    