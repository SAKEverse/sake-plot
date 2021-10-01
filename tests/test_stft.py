import pytest
import numpy as np
from stft import get_freq_index

# @pytest.fixture
# def freq_vector():
#     return np.arange(0, 100, 0.2)



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