
####----------------------- IMPORTS ------------------- ######
import pytest
import numpy as np
from stft import get_freq_index, Stft
####--------------------------------------------------- ######


####----------------------- Fixtures ------------------- ######
@pytest.fixture
def properties():
    prop = {'fs':4000, 'win_dur':5, 'freq_range': [5, 121], 
            'overlap':0.5, 'mains_noise': [58, 62]}
    return prop

@pytest.fixture
def fixed_frequency():
    def _method(index):
        freq = [5, 5 ,5, 5, 5, 5]
        return freq[index]
    return _method

@pytest.fixture
def frequency():
    def _method(index):
        freq = [5, 10 ,20, 60, 80, 100]
        return freq[index]
    return _method

@pytest.fixture
def ampltitude():
    def _method(index):
        freq = [20, 10 ,15, 5, 6, 3]
        return freq[index]
    return _method

@pytest.fixture
def simple_sine(): 
    def _method(properties, frequency, ampltitude, index):
        # Get x values of the sine wave
        time_duration = 30 # in seconds
        t = np.arange(0, time_duration, 1/properties['fs']);
        return np.sin(frequency(index)*t*np.pi*2) * ampltitude(index)
    return _method

@pytest.fixture
def noise_sine(): 
    def _method(properties, frequency, ampltitude, index):
        # Get x values of the sine wave
        time_duration = 30 # in seconds
        t = np.arange(0, time_duration, 1/properties['fs']);
        return np.sin(frequency(index)*t*np.pi*2) * ampltitude(index)
    return _method

@pytest.fixture
def mixed_sine(properties, frequency, ampltitude, index):

    # Get x values of the sine wave
    time_duration = 30 # in seconds
    t = np.arange(0, time_duration, 1/properties['fs']);
    
    # create sine waves
    y1 = np.sin(frequency(index[0])*t*np.pi*2) * ampltitude(index[0])
    y2 = np.sin(frequency(index[1])*t*np.pi*2) * ampltitude(index[1])   
    return y1 + y2


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
    Test whether inappropriate mains noise range inputs raise exception

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


@pytest.mark.parametrize("index", [(0), (1), (2), (3), (4)])    
def test_stft_simple_sine_freq(simple_sine, properties, frequency, ampltitude, index):
    """
    Test if correct peak frequency is detected from stft analysis

    Parameters
    ----------
    simple_sine : func, create sine wave
    properties : dict, with main properties
    frequency : func, get freqeuncy value based on index
    ampltitude : func, get amplitude value based on index
    index : int

    Returns
    -------
    None.

    """
    
    # init stft object
    stft_obj = Stft(properties)

    # get stft
    freq, pmat = stft_obj.get_stft(simple_sine(properties, frequency, ampltitude, index))
    
    # get psd
    psd = np.mean(pmat, axis = 1)
    
    # get peak frequency
    peak_freq = freq[np.argmax(psd)]
    
    assert peak_freq == frequency(index)


@pytest.mark.parametrize("index", [([0, 2]), ([3, 1]), ([5, 4]), ([1, 1])])    
def test_stft_simple_sine_amp(simple_sine, properties, fixed_frequency, ampltitude, index):
        
        power = []
        amp = []
        for i in index:
            # init stft object
            stft_obj = Stft(properties)
        
            # get stft
            freq_vector, pmat = stft_obj.get_stft(simple_sine(properties, fixed_frequency, ampltitude, i))
            
            # get psd
            psd = np.mean(pmat, axis = 1)
            
            # get frequency index to find peak power
            freq_idx = get_freq_index(freq_vector, [fixed_frequency(i)])
            
            # get power
            power.append(psd[freq_idx])
            amp.append(ampltitude(i))
        
        assert np.sign(power[0] - power[1]) == np.sign(amp[0] - amp[1])


def test_mains_noise(simple_sine, properties, frequency, ampltitude):
    
    index = 3 # for 60 Hz
    
    # init stft object
    stft_obj = Stft(properties)

    # get stft
    freq_vector, pmat = stft_obj.get_stft(simple_sine(properties, frequency, ampltitude, index))
    
    # get frequency index to find peak power
    freq_idx = get_freq_index(freq_vector, [frequency(index)])
    
    # get power before noise removal
    psd = np.mean(pmat, axis = 1)
    power_before_noise_removal =  psd[freq_idx]
    
    # remove surrounding frequency
    # freq = [frequency(index)-2, frequency(index)+2 ]
    pmat = stft_obj.remove_mains(freq_vector, pmat)
    
    # get power after noise removal
    psd = np.mean(pmat, axis = 1)
    power_after_noise_removal =  psd[freq_idx]    

    assert power_before_noise_removal > power_after_noise_removal













