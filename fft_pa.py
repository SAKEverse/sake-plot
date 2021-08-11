# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 15:01:11 2019

@author: panton01
"""

########## ------------------------------- IMPORTS ------------------------ ##########
import numpy as np
from beartype import beartype
from scipy import fft
from numba import jit, objmode
########## ---------------------------------------------------------------- ##########


# Single PSD class
class StftPa:
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
        
        # get frequency vector
        self.freq = self.freq_vec()
        
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
    def freq_vec(self) -> np.ndarray:
        """
        Get frequency vector

        Returns
        -------
        freq: np.ndarray, vector with frequencies that matches run_stft spectogram output

        """
               
        # create frequency array
        freq = np.arange(self.freq_range[0], self.freq_range[1]+(1/self.win_dur), 1/self.win_dur)
        
        return freq

@jit(nopython=True)
def rfft_numba(signal):
    # get normalised power
    with objmode(out = 'complex128[:]'):
            out = fft.rfft(signal, workers = 8)
    return out  
    
@jit(nopython=True)
def run_stft(input_wave, fs, winsize, overlap_size, f_idx):
    """
    Run short time fourier transfrom on input_wave.

    Parameters
    ----------
    input_wave : np.ndarray, 1D signal

    Returns
    -------
    power_matrix : 2D numpy array, rows = freq and columns = time bins

    """

    # trim signal if not divisible by window size
    signal_len = int(input_wave.shape[0] - (input_wave.shape[0]  % overlap_size))
    input_wave = input_wave[0:signal_len]

    # remove dc component
    input_wave = input_wave - np.mean(input_wave)
    
    # pad start and end
    input_wave = np.concatenate((input_wave[0: overlap_size], input_wave, input_wave[- overlap_size:]))
          
    # pre-allocate power matrix
    freq_bins = f_idx[1] - f_idx[0]+1
    time_bins = int((signal_len/overlap_size + 1))
    power_matrix = np.zeros((freq_bins, time_bins))
    
    # get fft normalization
    norm_var = (1/(fs*winsize))
    
    # initialise
    cntr = 0;
    
    # loop through signal segments with overlap 
    for i in range(0, signal_len + overlap_size, overlap_size):  
        
       # get segment
       signal = input_wave[i : i + winsize]
       
       # multiply the fft by hanning window
       signal = np.multiply(signal, np.hanning(winsize))
       
       # get normalised power
       xdft = rfft_numba(signal)
       xdft = np.multiply(np.square(np.absolute(xdft)), norm_var)
        
       # get normalised power spectral density
       power_matrix[:,cntr] = xdft[f_idx[0] : f_idx[1]+1]
       
       # update counter
       cntr+=1

    return power_matrix

    
    
    
    
    
    
    
    
    
    