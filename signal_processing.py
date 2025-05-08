from scipy.signal import welch, butter, filtfilt
from scipy.integrate import trapezoid
import numpy as np
import config

def butter_lowpass_filter(data, cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def butter_bandstop_filter(data, cutoff, fs, order=2):
    nyq = 0.5 * fs
    low = cutoff[0] / nyq
    high = cutoff[1] / nyq
    b, a = butter(order, [low, high], btype='bandstop')
    return filtfilt(b, a, data)

def moving_average(data, window=3):
    return np.convolve(data, np.ones(window)/window, mode='same')


def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)



def calculate_band_powers(freq, psd):
    band_powers = {}
    for band, (low, high) in config.FREQ_BANDS.items():
        mask = (freq >= low) & (freq <= high)
        band_powers[band] = trapezoid(psd[mask], freq[mask]) if sum(mask) > 0 else 0.0
    return band_powers