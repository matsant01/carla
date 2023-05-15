import scipy.fft as fft
import numpy as np


def esd(yt, fs):
    """Computes the Energy Spectral Density of a signal yt, keeping only positive frequencies (note that this means that the total energy is halved). 
    
    Parameters
    ----------
        yt : array_like 
            Signal to be analyzed
        fs : float 
            Sampling frequency of the signal

    Returns
    -------
        out : array_like, array_like
            Frequency axis and Energy Spectral Density of the signal
    """
    N = len(yt)
    f = fft.rfftfreq(N, 1/fs)
    Sxx = np.square(1 / fs) *  np.square(np.abs(fft.rfft(yt)))
    
    return f, Sxx