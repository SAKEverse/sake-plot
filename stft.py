# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 11:11:18 2021

@author: panton01
"""

########## ------------------------------- IMPORTS ------------------------ ##########
import numpy as np
import pandas as pd
from beartype import beartype
from scipy.signal import stft
########## ---------------------------------------------------------------- ##########

class GetIndex():
    "Get index"
    
    def __init__(self, array):
        self.array = array

    def find_nearest(self, value):
        """
        Find nearest value in self.array.

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
    Get frequency index.

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

def f_fill(arr:np.ndarray) -> np.ndarray:
    """
    Replace nans using pandas ffil method.

    Parameters
    ----------
    arr : np.ndarray

    Returns
    -------
    np.ndarray

    """
    df = pd.DataFrame(arr)
    df = df.fillna(method='ffill', axis=0)
    return  df.values

# Single PSD class
class Stft:
    """  
    Perform Stft analysis on 1D signal.

    """
    
    @beartype
    def __init__(self, fs:int, win_dur:int, freq_range:list, overlap:float = 0.5):
        """

        Parameters
        ----------
        fs : int
        win_dur : int
        freq_range : list
        overlap : float, the default is 0.5.

        Returns
        -------
        None.

        """

        # pass parameters to object
        self.fs = fs                                            # sampling rate (samples per second)
        self.win_dur = win_dur                                  # window duration (seconds)
        self.winsize = int(fs * win_dur)                        # window size (samples)  
        self.overlap = overlap                                  # overlap (ratio)
        self.overlap_size = int(self.winsize * self.overlap)    # overlap size (samples)
        self.freq_range = freq_range                            # frequency range (Hz)
        
        # get frequency index
        self.f_idx = self.get_freq_idx(self.freq_range)
    
    @beartype
    def get_freq_idx(self, f:list) -> np.ndarray:
        """
        Convert frequency value to frequency index based on sampling rate

        Parameters
        ----------
        f : list, containing frequency value

        Returns
        -------
        freq_idx : list, frequency index value(int)

        """
        
        freq_idx = np.zeros(len(f), dtype = np.int32)
        for i in range(len(f)): 
            freq_idx[i] = int(f[i]*(self.winsize/self.fs))
        return freq_idx
        
    @beartype
    def get_stft(self, input_wave:np.ndarray):
        """
        Run short time fourier transfrom on input_wave.

        Parameters
        ----------
        input_wave : np.ndarray, 1D signal

        Returns
        -------
        f: 1D, frequency values
        power_matrix : 2D numpy array, rows = freq and columns = time bins

        """
        
        f, t, pmat = stft(input_wave, self.fs, nperseg=self.winsize, noverlap = self.overlap_size)
        pmat = np.square(np.abs(pmat[self.f_idx[0] : self.f_idx[1]+1,:]))
        
        return f[self.f_idx[0] : self.f_idx[1]+1], pmat

    @staticmethod
    def remove_mains(freq:np.ndarray, pmat:np.ndarray, f_noise:list) -> np.ndarray:
        """
        Remove mains noise, using nans replacement and

        Parameters
        ----------
        freq : np.ndarray
        pmat : np.ndarray
        f_noise : list

        Returns
        -------
        pmat : np.ndarray

        """

        # find frequency index
        f_idx = get_freq_index(freq, f_noise)

        # set noise index to NaNs
        pmat[f_idx[0]:f_idx[1]+1,:] = np.nan

        # fill NaNs
        pmat = f_fill(pmat)

        return pmat
    
    
    ### ADD OUTLIER REMOVAL ###


    def run_stft(self, input_wave:np.ndarray, f_noise:list):
        """
        Get stft and remove mains noise.

        Parameters
        ----------
        input_wave : np.ndarray, 1D signal
        f_noise : list, lower and upper bounds of mains noise

        Returns
        -------
        freq : np.ndarray, real frequency vector
        pmat : np.ndarray, transformed spectogram

        """

        # get stft
        freq, pmat = self.get_stft(input_wave)

        # remove mains nose
        pmat = Stft.remove_mains(freq, pmat, f_noise)

        return freq, pmat
        
        
        
        
        
        
        
        
        
        
        
        
        