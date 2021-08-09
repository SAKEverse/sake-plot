# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 13:24:36 2021

@author: panton01
"""

########## ------------------------------- IMPORTS ------------------------ ##########
import numpy as np
import pandas as pd
from fft_pa import Stft
from get_data import AdiGet
from filter_index import load_n_filter
from beartype import beartype
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
def get_power_area(stft_obj, freqs:np.ndarray, signal:np.ndarray) -> np.ndarray:
    """
    Get power area across frequencies

    Parameters
    ----------
    stft_obj : stft obj
    freqs : np.ndarray, 2D array with frequencies, rows = different frequency ranges, colums = [start, stop]

    Returns
    -------
    powers : 1D np.array, len = frequency ranges 

    """
    
    # get frequency index
    freq_idx = get_freq_index(stft_obj.freq, freqs)
    
    # get power matrix
    pmat = stft_obj.run_stft(signal)
    
    # init empty array to store powers
    powers = np.zeros(freq_idx.shape[0])
    for i in range(freq_idx.shape[0]):
        powers[i] = np.sum(pmat[freq_idx[i,0]:freq_idx[i,1],:])
    
    return powers
        

def add_powers(index_df):
    
    fft_duration = 5
    freq_range = [0.2, 120]
    freqs = np.array([[0.2,5], [6,12], [15,30], [31,70], [80,120]])
    
    # create column names
    col_names = []
    for i in range(freqs.shape[0]):
        col_names.append(str(freqs[i,0]) + ' - ' + str(freqs[i,1]) + ' Hz')
        
    # create array for storage
    power_array = np.empty((len(index_df), freqs.shape[0]))
    for i in range(len(index_df)): # iterate over dataframe
        
        # get properties
        properties = index_df[AdiGet.input_parameters].loc[i].to_dict()
        
        # get signal
        signal = AdiGet(properties).get_data_adi()
    
        # run PSD
        stft_obj =  Stft(int(index_df['sampling_rate'][i]), fft_duration, freq_range)
        
        # get power matrix 
        power_array[i,:] = get_power_area(stft_obj, freqs, signal)
    
    # concatenate to array
    index_df = pd.concat([index_df, pd.DataFrame(data = power_array, columns = col_names)], axis=1)
    
    return index_df
        


if __name__ == '__main__':
    
    # # define path and conditions for filtering
    # path = r'C:\Users\panton01\Desktop\index.csv'
    # filter_conditions = {'brain_region':['bla'], 'treatment':['baseline','vehicle']}
    
    # # filter index based on conditions
    # index_df = load_n_filter(path, filter_conditions)
    
    # # add powers
    # index_df = add_powers(index_df)
    














