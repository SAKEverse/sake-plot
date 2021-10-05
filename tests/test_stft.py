import pytest
import numpy as np
from stft import get_freq_index, Stft

@pytest.fixture
def properties():
    prop = {'fs':4000, 'win_dur':5, 'freq_range': [1, 121], 
            'overlap':0.5, 'mains_noise': [50, 100]}
    return prop

@pytest.fixture
def freq():
    return 5

@pytest.fixture
def input_wave(properties, freq):
    # Get x values of the sine wave
    time_duration = 30 # in seconds
    t = np.arange(0, time_duration, 1/properties['fs']);
    return np.sin(freq*t*np.pi*2)



@pytest.mark.parametrize("freq_range", [ ([1, 10 ,200]),
                                          ([1, '10']),
                                          ([-10, 100]),
                                          ([100, 10]),
                                          ([1, 3000]),
                                          ])
def test_freq_range(properties:dict, freq_range:list):
    """
    # Test whether inappropriate frequency range inputs raise exception

    Parameters
    ----------
    properties : dict, with properties
    freq_range : list, with tests for frequency range input
    """
    
    # pass inputs to properties dictionary
    properties.update({'freq_range': freq_range})
    
    with pytest.raises(Exception):
        # init stft object
        Stft(properties)

    
def test_stft_transform(input_wave, properties):
    
        # init stft object
        stft_obj = Stft(properties)
    
        # get stft
        freq, pmat = stft_obj.get_stft(input_wave)
    
        assert freq.shape[0] == pmat.shape[0]







@pytest.mark.parametrize("freq_vector, input_freqs, test_freqs", 
                          [(np.arange(0, 100, 0.2), [8], [40]), 
                          (np.arange(0, 100, 2), [8], [4]),
                          (np.arange(3, 100, 3), [5], [1]),
                          (np.arange(4, 100, 0.2), [6], [10]),
                          (np.arange(2, 100, 1), [10, 42], [8, 40])
                                                ])
def test_get_freq_index(freq_vector, input_freqs, test_freqs):
    """
    Test if the correct frequency index is obtained from frequency values
    
    Parameters
    ----------
    freq_vector : np.ndarray, frequency vector to be indexed
    input_freqs : values to find the index
    test_freqs : ground truth values

    Returns
    -------
    None.

    """
    
    freq_idx = get_freq_index(freq_vector, input_freqs)
    
    assert np.all(test_freqs == freq_idx) 