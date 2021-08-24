# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 10:09:35 2021

@author: gweiss01
"""

import numpy as np
from numba import njit


@njit
def get_threshold(vector:np.ndarray, threshold:float) -> dict:
    """
    Find positive and negative thresholds

    Parameters
    ----------
    vector : np.ndarray
    threshold : float, multiplier for threshold

    Returns
    -------
    nnp.ndarray: [negative,positive] outlier threshold

    """
    
    # find positive and outlier threshold
    outlier_threshold_pos = (np.median(vector) + (np.std(vector)*threshold))
    
    # find negative and outlier threshold
    outlier_threshold_neg = (np.median(vector) - (np.std(vector)*threshold))
    
    return np.array( [outlier_threshold_neg, outlier_threshold_pos] ) #{'pos' : outlier_threshold_pos, 'neg': outlier_threshold_neg}

@njit
def rolling_outliers(arr:np.ndarray, window:int, threshold:float) -> np.ndarray:
    """
    Find outliers from vector

    Parameters
    ----------
    time_vector : np.ndarray
    window : int
    threshold : float

    Returns
    -------
    outliers : np.ndarray

    """
    
    # create vector and convert to bool
    outliers = np.zeros(arr.shape)
    outliers = outliers == 1
    
    # half window
    half_win = int(np.ceil(window/2))
    
    ## start
    # get baseline and threshold
    baseline = arr[:half_win]
    outlier_threshold = get_threshold(baseline, threshold)

    for base_cnt,i in enumerate(range(half_win)):
        # find outliers
        outliers[i] = (baseline[base_cnt] < outlier_threshold[0]) | (baseline[base_cnt] > outlier_threshold[1])
    
    ## middle  
    # get baseline and threshold
    for i in range(i+1, arr.shape[0] - half_win):
        
        # get baseline and threshold
        baseline = arr[i - half_win : i + half_win]  
        outlier_threshold = get_threshold(baseline, threshold)
        
        # find outliers
        outliers[i] = (baseline[half_win] < outlier_threshold[0]) | (baseline[half_win] > outlier_threshold[1])
    
    ## end
    # get baseline and threshold
    baseline = arr[i+1:]
    outlier_threshold = get_threshold(baseline, threshold)
    
    for base_cnt,i in enumerate(range(i+1, arr.shape[0])):
        # find outliers
        outliers[i] = (baseline[base_cnt] < outlier_threshold[0]) | (baseline[base_cnt] > outlier_threshold[1])

    return outliers


def median_outliers(arr:np.ndarray, window:int, threshold:float) -> np.ndarray:
    '''
    finds the outliers based on median by reshaping

    Parameters
    ----------
    arr : np.ndarray
    window : int
    threshold : float

    Returns
    -------
    outliers : np.ndarray

    '''

    # first make a threshold index:
        
    # divide the array into windows by reshaping, then find the median of each window, this makes an array of size: original_len/window    
    medians = np.median(arr[:(arr.shape[0]//window)*window].reshape(-1,window), axis=1)
    
    # expand the array back to full size by interpolaing the in between values
    smoothed_medians = np.interp(np.linspace(0, medians.shape[0], medians.shape[0]*window), np.arange(medians.shape[0]), medians)
    
    # fill the end values 
    time_thresholds = np.append(smoothed_medians,[smoothed_medians[-1]]*(arr.shape[0]-smoothed_medians.shape[0]))
    
    # compare the original array to the threshold index to find the outliers
    outliers = ((arr>(time_thresholds*threshold)) |  (arr<-(time_thresholds*threshold)))
    
    return outliers


def std_outliers(arr:np.ndarray, window:int, threshold:float) -> np.ndarray:
    '''
    Find outliers based on standard deviation, using pandas

    Parameters
    ----------
    arr : np.ndarray
    window : int
    threshold : float

    Returns
    -------
    outliers : np.ndarray

    '''
    
    #first make a threshold index:
        
    time_thresholds=pd.Series(arr).rolling(window).std().fillna(method='bfill')
    
    #compare the original array to the threshold index to find the outliers
    
    outliers = ((arr>(time_thresholds*threshold)) |  (arr<-(time_thresholds*threshold)))
    
    return outliers


def mean_outliers(arr:np.ndarray, window:int, threshold:float) -> np.ndarray:
    '''
    Find outliers based on mean, using numpy convolve

    Parameters
    ----------
    arr : np.ndarray
    window : int
    threshold : float

    Returns
    -------
    outliers : np.ndarray

    '''
    
    #first make a threshold index:
        
    time_thresholds=np.convolve(arr, np.ones(window)/window, mode='valid')
    
    #forward fill the terminal values
    
    time_thresholds=np.append(time_thresholds,[time_thresholds[-1]]*(len(arr)-len(time_thresholds)))
    
    #compare the original array to the threshold index to find the outliers
    
    outliers = ((arr>(time_thresholds*threshold)) |  (arr<-(time_thresholds*threshold)))
    
    return outliers


# define function that will be executed
get_outliers = rolling_outliers







