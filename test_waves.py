# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 16:56:40 2021

@author: panton01
"""
import numpy as np
import matplotlib.pyplot as plt
from time import time
from stft import Stft
# from numba import njit, objmode
# from scipy import fft
from fft_pa import StftPa, run_stft

fs = 1000
f = 10.2
t_seconds = 100*120
nloops = 10
freq_range = [1, 150]

t = np.arange(0,t_seconds,1/fs)
signal = np.sin(2 * np.pi * f * t)
signal = signal * np.linspace(1, 1.5, signal.shape[0])
# plt.plot(t, signal)



# pa implementation
stft_objpa = StftPa(fs, 5, freq_range)
fs = stft_objpa.fs
winsize = stft_objpa.winsize
overlap_size = stft_objpa.overlap_size
f_idx = stft_objpa.f_idx

tic = time()
for i in range(nloops):
    pmat = run_stft(signal, fs, winsize, overlap_size, f_idx)
print('Time elapsed PA:', time()-tic, 'seconds')
# plt.plot(stft_obj.freq, np.mean(pmat, axis=1))


stft_obj =  Stft(fs, 5, freq_range)
tic = time()
for i in range(nloops):
    f,pmat = stft_obj.run_stft(signal)
print('Time elapsed scipy:', time()-tic, 'seconds')
# plt.plot(f, np.mean(pmat, axis=1))



# @njit
# def fft_numba(signal):
#         # get normalised power
#         with objmode(out = 'complex128[:]'):
#                 out = fft.rfft(signal)
#         return out
    
        
        
# @njit
# def fft_loop(signal):
#     a = np.zeros(nloops)
#     for i in range(nloops):
#         a[i] = fft_numba(signal)
#     return a
    

# tic = time()
# # with fft.set_workers(8):
# fft_numba(signal)
# print('Time elapsed scipy:', time()-tic, 'seconds')



















