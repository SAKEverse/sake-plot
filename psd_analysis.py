# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 13:24:36 2021

@author: panton01
"""

########## ------------------------------- IMPORTS ------------------------ ##########
import os
import numpy as np
import pandas as pd
from stft import Stft
from get_data import AdiGet
from filter_index import load_n_filter
from beartype import beartype
from typing import TypeVar, Union
PandasDf = TypeVar('pandas.core.frame.DataFrame')
########## ---------------------------------------------------------------- ##########

class GetIndex():
    "Get index"
    
    def __init__(self, array):
        self.array = array

    def find_nearest(self, value):
        """
        find nearest value in self.array

        Parameters
        ----------
        value : values to search the array for

        Returns
        -------
        idx for values
        
        """
        return (np.abs(self.array - value)).argmin()

@beartype
def get_freq_index(freq_vector:np.ndarray, freqs) -> np.ndarray:
    """
    Get frequency index

    Parameters
    ----------
    freq_vector : np.ndarray, frequency vector to be indexed
    freqs : values to find the index

    Returns
    -------
    np.ndarray

    """
    
    # instantiate
    f = GetIndex(freq_vector)
    
    # vectorize function
    vfunc = np.vectorize(f.find_nearest)

    # get index 
    return vfunc(freqs)

@beartype
def get_power_area(pmat:np.ndarray, freq_vec:np.ndarray, freqs:np.ndarray) -> np.ndarray:
    """
    Get power area across frequencies

    Parameters
    ----------
    pmat : np.ndarray, 2D array containing power values rows = frequency bins and cols = time bins
    freq_vec : np.ndarray, vector with real frequency values
    freqs : np.ndarray, 2D array with frequencies, rows = different frequency ranges, colums = [start, stop]

    Returns
    -------
    powers : 1D np.array, len = frequency ranges 

    """
    
    # get frequency index
    freq_idx = get_freq_index(freq_vec, freqs)
    
    # init empty array to store powers
    powers = np.zeros(freq_idx.shape[0])
    for i in range(freq_idx.shape[0]):
        powers[i] = np.sum(pmat[freq_idx[i,0]:freq_idx[i,1],:])
    
    return powers
        
@beartype
def get_pmat(index_df:PandasDf, fft_duration:int = 5, freq_range:list = [1, 120]) -> PandasDf:
    """
    Run Stft analysis on signals retrieved using rows of index_df

    Parameters
    ----------
    index_df : PandasDf
    fft_duration : int, The default is 5.
    freq_range : list, The default is [1, 120].

    Returns
    -------
    PandasDf, with power matrix and frequency

    """

    # create empty series dataframe
    df = pd.DataFrame(np.empty((len(index_df), 2)), columns = ['pmat', 'freq'], dtype = object)

    for i in range(len(index_df)): # iterate over dataframe
        
        # get properties
        properties = index_df[AdiGet.input_parameters].loc[i].to_dict()
        
        # get signal
        signal = AdiGet(properties).get_data_adi()
           
        # run PSD
        stft_obj =  Stft(int(index_df['sampling_rate'][i]), fft_duration, freq_range)

        # get frequency vector and power matrix 
        df.at[i, 'freq'], df.at[i, 'pmat'] = stft_obj.run_stft(signal)
    
    return df


def melted_power_area(index_df:PandasDf, power_df:PandasDf, freqs:list, selected_categories:list):
    """
    Get power area and melt dataframe for seaborn plotting.

    Parameters
    ----------
    index_df : PandasDf, experiment index
    power_df : PandasDf, contains pmat and frequency vectors for every row of index_df
    freqs : list, 2D list with frequency ranges for extraction of power area
    selected_categories : list, columns that will be included in the melted

    Returns
    -------
    df : PandasDf, melted df with power area and categories

    """
    
    # create frequency column names
    freq_columns = []
    for i in range(freqs.shape[0]):
        freq_columns.append(str(freqs[i,0]) + ' - ' + str(freqs[i,1]) + ' Hz')
    
    # create array for storage
    power_array = np.empty((len(index_df), freqs.shape[0]))
    for i in range(len(index_df)): # iterate over dataframe
        
        # get power across frequencies
        power_array[i,:] = get_power_area(power_df['pmat'][i], power_df['freq'][i], freqs)
        
    # concatenate to array
    index_df = pd.concat([index_df, pd.DataFrame(data = power_array, columns = freq_columns)], axis=1)
        
    # melt dataframe for seaborn plotting
    df = pd.melt(index_df, id_vars = selected_categories, value_vars = freq_columns, var_name = 'freq', value_name = 'power_area')
    
    return df



if __name__ == '__main__':
    
    ### ---------------------- USER INPUT -------------------------------- ###
    
    # define path and conditions for filtering
    filename = 'index.csv'
    parent_folder = r'C:\Users\panton01\Desktop\pydsp_analysis'
    path =  os.path.join(parent_folder, filename)
    
    # enter filter conditions
    filter_conditions = {'brain_region':['bla', 'pfc'], 'treatment':['baseline','vehicle']}
    
    # define frequencies of interest
    freqs = np.array([[2,5], [6,12], [15,30], [31,70], [80,120]])
    
    #### ---------------------------------------------------------------- ####
    
    # filter index based on conditions
    index_df = load_n_filter(path, filter_conditions)
    
    # save dataframe
    index_df.to_pickle(os.path.join(parent_folder, filename.replace('csv','pickle')))
    
    # get pmat
    power_df = get_pmat(index_df)
    power_df.to_pickle(os.path.join(parent_folder, 'power_' + filename.replace('csv','pickle')))
    
    # get power area across frequencies
    df = melted_power_area(index_df, power_df, freqs, ['sex', 'treatment', 'brain_region'])
    df.to_csv(os.path.join(parent_folder,  'melt_' + filename), index= False)

    import seaborn as sns
    sns.catplot(data = df, x = 'freq', y = 'power_area', hue = 'treatment', col = 'sex', row = 'brain_region', kind = 'box')













