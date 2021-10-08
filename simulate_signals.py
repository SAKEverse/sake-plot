####----------------------- IMPORTS ------------------- ######
import numpy as np
from stft import get_freq_index, Stft
from matplotlib.pyplot import plot, hist
####--------------------------------------------------- ######

# define parameters
properties = {'fs':1000, 'win_dur':5, 'freq_range': [5, 121], 
            'overlap':0.5, 'mains_noise': [58, 62]}
freq = 60
fs = properties['fs']

# create time vector
signal_time = 60
t = np.arange(0, signal_time + (1/fs), 1/fs)

# create template wave
template = np.sin(freq*t[0:int(np.ceil(fs/freq))+1]*np.pi*2)

# create normaly distributed events
mu = 1/freq
sigma = mu/1000
n_events = int(np.ceil(t.shape[0]/template.shape[0]))
s = np.random.normal(mu, sigma, n_events+100)

# get inter event interval and find index
isi = np.cumsum(s)
index = get_freq_index(t, isi)

# create logic vector to be convolved
logic_vector = np.zeros(t.shape)
logic_vector[index] = 1

# create convolved signal
signal = np.convolve(logic_vector, template, mode = 'same')

plot(signal)
# hist(index, bins = 50)

# get power
# stft_obj = Stft(properties)
# freq_vector, pmat = stft_obj.get_stft(signal)
# psd = np.mean(pmat, axis = 1)
# plot(freq_vector, psd)













