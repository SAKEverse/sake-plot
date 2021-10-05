
####----------------------- IMPORTS ------------------- ######
import pytest
import numpy as np
from stft import get_freq_index, Stft
####--------------------------------------------------- ######


####----------------------- Fixtures ------------------- ######
@pytest.fixture
def properties():
    prop = {'fs':4000, 'win_dur':5, 'freq_range': [5, 121], 
            'overlap':0.5, 'mains_noise': [50, 100]}
    return prop

@pytest.fixture
def frequency():
    return 5

@pytest.fixture
def simple_sine(properties, frequency):
    # Get x values of the sine wave
    time_duration = 30 # in seconds
    t = np.arange(0, time_duration, 1/properties['fs']);
    return np.sin(frequency*t*np.pi*2)
####--------------------------------------------------- ######


####---------------------------- Tests -------------------------- ######
@pytest.mark.parametrize("freq_vector, input_freqs", 
                          [(np.arange(0, 100, 0.2), [8, 'test']), 
                          (np.arange(0, 100, 2), [1, '2', 3]),
                                                ])
def test_get_freq_index(freq_vector, input_freqs):
    """
    Test if function raises exception for incorrect input types
    
    Parameters
    ----------
    freq_vector : np.ndarray, frequency vector to be indexed
    input_freqs : values to find the index
    test_freqs : ground truth values

    Returns
    -------
    None.

    """
    with pytest.raises(Exception):
        get_freq_index(freq_vector, input_freqs)


@pytest.mark.parametrize("freq_vector, input_freqs, test_freqs", 
                          [(np.arange(0, 100, 0.2), [8], [40]), 
                          (np.arange(0, 100, 2), [8], [4]),
                          (np.arange(3, 100, 3), [5], [1]),
                          (np.arange(4, 100, 0.2), [6], [10]),
                          (np.arange(2, 100, 1), [10, 42], [8, 40])
                                                ])
def test_get_freq_index_output(freq_vector, input_freqs, test_freqs):
    """
    Test if function raises exception for incorrect input types
    
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


@pytest.mark.parametrize("freq_range", [ ([1, 10 ,200]),
                                          ([1, '10']),
                                          ([-10, 100]),
                                          ([100, 10]),
                                          ([1, 3000]),
                                          ])
def test_freq_range(properties:dict, freq_range:list):
    """
    Test if function raises exception for incorrect input types

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
        
@pytest.mark.parametrize("mains_noise", [ ([1, 10 ,200]),
                                          ([1, '10']),
                                          ([3, 100]),
                                          ([100, 10]),
                                          ([1, 200]),
                                          ])
def test_mains_noise_range(properties:dict, mains_noise:list):
    """
    # Test whether inappropriate mains noise range inputs raise exception

    Parameters
    ----------
    properties : dict, with properties
    freq_range : list, with tests for frequency range input
    """
    
    # pass inputs to properties dictionary
    properties.update({'mains_noise': mains_noise})
    
    with pytest.raises(Exception):
        # init stft object
        Stft(properties)

    
def test_stft_transform(simple_sine, properties, frequency):
    
        # init stft object
        stft_obj = Stft(properties)
    
        # get stft
        freq, pmat = stft_obj.get_stft(simple_sine)
        
        # get psd
        psd = np.mean(pmat,axis=1)
        
        # get peak frequency
        peak_freq = freq[np.argmax(psd)]
        
        assert peak_freq == frequency



















