# This module is for the thresholding of the MEA signal
import numpy as np
import matplotlib.pyplot as plt
from modules.utils import detect_threshold_crossings, align_to_minimum, extract_waveforms, find_spike_peaks
from modules.visualization import plot_processed_signal, plot_waveforms

def thresholding_voltage(signal, fs, electrode_stream): #TODO: omit from electrode stream and just plot the filtered signal
    '''Function to threshold signal based on voltage amplitude
    Inputs
        signal = signal
        fs: sampling frequency'''
    noise_std = np.std(signal)
    noise_mad = np.median(np.absolute(signal)) / 0.6745
    print('Noise Estimate by Standard Deviation: {0:g} V'.format(noise_std))
    print('Noise Estimate by MAD Estimator     : {0:g} V'.format(noise_mad))

    spike_threshold = -3.75 * noise_mad # roughly -30 µV
    plt.plot(signal)
    # Adding a horizontal line at y=5
    plt.axhline(y=spike_threshold, color='r', linestyle='--', label='threshold')
    plt.savefig(f'./imgs/Thresholding.png')
    plt.show()


    crossings = detect_threshold_crossings(signal, fs, spike_threshold, 0.003) # dead time of 3 ms
    spks = align_to_minimum(signal, fs, crossings, 0.002) # search range 2 ms



    # Align to minimum
    
    timestamps = spks / fs
    range_in_s = (0, 120)
    spikes_in_range = timestamps[(timestamps >= range_in_s[0]) & (timestamps <= range_in_s[1])]
    pre = 0.001# 1 ms
    post= 0.002# 2 ms

    cutouts = extract_waveforms(signal, fs, spks, pre, post)
    print("Cutout array shape: " + str(cutouts.shape)) # number of spikes x number of samples
    print("First few spks:", spks[:10])

    plot_waveforms(cutouts,fs,pre,post,10)
    return spikes_in_range

