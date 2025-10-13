from scipy import signal
import numpy as np

def bandpass_filter(x, fs, lo, hi, order=3):
    sos = signal.butter(order, [lo, hi], btype='bandpass', fs=fs, output='sos')
    return signal.sosfiltfilt(sos, x, axis=0)

def lowpass_filter(x, fs, cutoff, order=4):
    sos = signal.butter(order, cutoff, btype='lowpass', fs=fs, output='sos')
    return signal.sosfiltfilt(sos, x, axis=0)

def notch_filter(x, fs, freq=50.0, q=30):
    if freq <= 0:
        return x
    b, a = signal.iirnotch(w0=freq, Q=q, fs=fs)
    return signal.filtfilt(b, a, x, axis=0)

def mad(x):
    return np.median(np.abs(x - np.median(x))) / 0.6745