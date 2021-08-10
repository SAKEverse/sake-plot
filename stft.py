# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 11:11:18 2021

@author: panton01
"""

########## ------------------------------- IMPORTS ------------------------ ##########
import numpy as np
from beartype import beartype
from scipy.signal import stft
########## ---------------------------------------------------------------- ##########



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
    def run_stft(self, input_wave:np.ndarray):
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
        
        
        
        
        
        
        
        
        
        
        
        
        