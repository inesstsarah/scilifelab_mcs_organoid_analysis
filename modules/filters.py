from scipy import signal
import numpy as np
import spkit as sp
import pywt
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

#def wavelet_filter(x)

def wavelet_filter(signal, wavelet="db4", level=None, mode="soft", threshold_scale=1.0):
    """
    Wavelet-based denoising for 1D signals.

    Parameters
    ----------
    signal : array-like
        Input 1D signal.
    wavelet : str
        Wavelet family name (e.g., "db4", "sym5", "coif3").
    level : int or None
        Decomposition level. If None, pywt will choose the maximum possible.
    mode : str
        Thresholding mode: "soft" or "hard".
    threshold_scale : float
        Multiplier for the universal threshold.

    Returns
    -------
    denoised : np.ndarray
        The filtered signal.
    """

    # Wavelet decomposition
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # Estimate noise from the detail coefficients at the highest level
    detail_coeffs = coeffs[-1]
    # 75th percentile of normal distribution
    sigma = np.median(np.abs(detail_coeffs)) / 0.6745

    # Universal threshold
    threshold = threshold_scale * sigma * np.sqrt(2 * np.log(len(signal)))

    # Apply thresholding to detail coefficients
    new_coeffs = [coeffs[0]]  # keep approximation untouched
    for c in coeffs[1:]:
        new_coeffs.append(pywt.threshold(c, threshold, mode=mode))

    # Reconstruct signal
    denoised = pywt.waverec(new_coeffs, wavelet)

    return denoised
