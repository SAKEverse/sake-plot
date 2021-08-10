# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 13:24:36 2021

@author: panton01
"""

########## ------------------------------- IMPORTS ------------------------ ##########
import numpy as np
import pandas as pd
from stft import Stft
from get_data import AdiGet
from filter_index import load_n_filter
from beartype import beartype
from typing import TypeVar
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
def add_pmat(index_df:PandasDf, fft_duration:int = 5, freq_range:list = [1, 120]) -> PandasDf:
    """
    Add power matrix and frequency vector for each row to index dataframe.

    Parameters
    ----------
    index_df : PandasDf
    fft_duration : int, The default is 5.
    freq_range : list, The default is [1, 120].

    Returns
    -------
    PandasDf

    """

    
    # add empty series to dataframe
    index_df['pmat'] = ''
    index_df['freq'] = ''
    
    for i in range(len(index_df)): # iterate over dataframe
        
        # get properties
        properties = index_df[AdiGet.input_parameters].loc[i].to_dict()
        
        # get signal
        signal = AdiGet(properties).get_data_adi()
           
        # run PSD
        stft_obj =  Stft(int(index_df['sampling_rate'][i]), fft_duration, freq_range)

        # get frequency vector and power matrix 
        index_df.at[i, 'freq'], index_df.at[i, 'pmat'] = stft_obj.run_stft(signal)
    
    return index_df


def add_power_area(index_df:PandasDf, freqs:np.ndarray):
    """
    

    Parameters
    ----------
    index_df : TYPE
        DESCRIPTION.
    freqs : TYPE
        DESCRIPTION.

    Returns
    -------
    index_df : TYPE
        DESCRIPTION.
    col_names : TYPE
        DESCRIPTION.

    """
    
    # create column names
    col_names = []
    for i in range(freqs.shape[0]):
        col_names.append(str(freqs[i,0]) + ' - ' + str(freqs[i,1]) + ' Hz')
    
    # create array for storage
    power_array = np.empty((len(index_df), freqs.shape[0]))
    for i in range(len(index_df)): # iterate over dataframe
        
        # get power across frequencies
        power_array[i,:] = get_power_area(index_df['pmat'][i], index_df['freq'][i], freqs)
     
    # concatenate to array
    index_df = pd.concat([index_df, pd.DataFrame(data = power_array, columns = col_names)], axis=1)
    
    return index_df, col_names



if __name__ == '__main__':
    
    # define path and conditions for filtering
    path = r'C:\Users\panton01\Desktop\index.csv'
    filter_conditions = {'brain_region':['bla', 'pfc'], 'treatment':['baseline','vehicle']}
    
    # # filter index based on conditions
    index_df = load_n_filter(path, filter_conditions)
    
    # # add powers
    freqs = np.array([[2,5], [6,12], [15,30], [31,70], [80,120]])
       
    # add pmat to index_df
    index_df = add_pmat(index_df)
    
    # add power area across frequencies
    index_df, parea_freqs = add_power_area(index_df, freqs)
    
    # save dataframe
    index_df.to_pickle("./index.pkl")
    
    # import seaborn as sns
    df = index_df.drop(['freq', 'pmat'], axis=1)
    df = pd.melt(df, id_vars=['sex', 'treatment', 'brain_region'], value_vars = parea_freqs)
    
    import seaborn as sns
    sns.catplot(data = df, x = 'variable', y = 'value', hue = 'treatment', col = 'sex', row = 'brain_region', kind = 'box')













