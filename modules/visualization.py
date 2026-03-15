
# ---- IMPORTS ----

import matplotlib.pyplot as plt
# MCS PyData tools
import McsPy
import McsPy.McsData
from McsPy import ureg, Q_
import numpy as np

def plot_signal(signal, fs, title, dur):
    plt.plot(signal)
    duration = dur
    t = np.arange(0, duration, 1/fs)
    plt.figure(figsize=(10, 4))
    plt.plot(t, signal, color='blue', linewidth=2)
    plt.title(f'{title}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude (V)')
    plt.grid()
    plt.xlim([0, duration])
    plt.savefig(f'./imgs/{title}.png')

    plt.show()

def plot_processed_signal(signal, filtered_signal, fs, spikes_in_range, dur):
    t = np.arange(0, dur, 1/fs)
    
    plt.figure(figsize=(10, 4))

    plt.plot(t,signal, label = "Original Signal (V)")

    plt.plot(t,filtered_signal, label='Filtered Signal (V)')

    plt.plot(spikes_in_range, [0]*spikes_in_range.shape[0], 'ro', ms=2, label = "Detected Spikes")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.title("Detected Spikes")
    plt.legend(loc="right")
    plt.savefig(f'./imgs/Processed Signal.png')
    plt.show()

def plot_GMM_cluters(cutouts, labels, n_components,pre,post, fs):
        '''Plot GMM clusters on cutout data'''
        _ = plt.figure(figsize=(8,8))
        for i in range(n_components):
            idx = labels == i
            color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i]
            plot_waveforms(cutouts[idx,:], fs, pre, post, n=100, color=color, show=False)
        plt.show()

def plot_mean_waveform(cutouts, fs):
    t = np.arange(0, 0.004, 1/fs)

    mean_waveform = np.mean(cutouts, axis=0)
    plt.plot(mean_waveform)
    plt.title("Mean Waveform")
    plt.savefig("./imgs/Mean Waveform.png")
    plt.show()

def plot_spike_raster(spikes_in_range):
    # plot spike train
    plt.plot(spikes_in_range, np.ones_like(spikes_in_range), '|', markersize=100, color='black')
    plt.title('Output raster')
    plt.xlabel('Time (s)')
    plt.savefig(f'./imgs/Spike Raster.png')
    plt.show()

def plot_analog_stream_channel(analog_stream, channel_idx, from_in_s=0, to_in_s=None, show=True):
    """
    Plots data from a single AnalogStream channel
    
    :param analog_stream: A AnalogStream object
    :param channel_idx: A scalar channel index (0 <= channel_idx < # channels in the AnalogStream)
    :param from_in_s: The start timestamp of the plot (0 <= from_in_s < to_in_s). Default: 0
    :param to_in_s: The end timestamp of the plot (from_in_s < to_in_s <= duration). Default: None (= recording duration)
    :param show: If True (default), the plot is directly created. For further plotting, use show=False
    """
    # extract basic information
    ids = [c.channel_id for c in analog_stream.channel_infos.values()]
    channel_id = ids[channel_idx]
    channel_info = analog_stream.channel_infos[channel_id]
    sampling_frequency = channel_info.sampling_frequency.magnitude
   
    # get start and end index
    from_idx = max(0, int(from_in_s * sampling_frequency))
    if to_in_s is None:
        to_idx = analog_stream.channel_data.shape[1]
    else:
        to_idx = min(analog_stream.channel_data.shape[1], int(to_in_s * sampling_frequency))
        
    # get the timestamps for each sample
    time = analog_stream.get_channel_sample_timestamps(channel_id, from_idx, to_idx)

    # scale time to seconds:
    scale_factor_for_second = Q_(1,time[1]).to(ureg.s).magnitude
    time_in_sec = time[0] * scale_factor_for_second
    
    # get the signal
    signal = analog_stream.get_channel_in_range(channel_id, from_idx, to_idx)

    # scale signal to µV:
    scale_factor_for_uV = Q_(1,signal[1]).to(ureg.uV).magnitude
    signal_in_uV = signal[0] * scale_factor_for_uV

    # construct the plot
    _ = plt.figure(figsize=(20,6))
    _ = plt.plot(time_in_sec, signal_in_uV)
    _ = plt.xlabel('Time (%s)' % ureg.s)
    _ = plt.ylabel('Voltage (%s)' % ureg.uV)
    _ = plt.title('Channel %s' % channel_info.info['Label'])
    if show:
        plt.show()


def plot_waveforms(cutouts, fs, pre, post, n=100, color='k', show=True):
    """
    Plot an overlay of spike cutouts
    
    :param cutouts: A spikes x samples array of cutouts
    :param fs: The sampling frequency in Hz
    :param pre: The duration of the cutout before the spike in seconds
    :param post: The duration of the cutout after the spike in seconds
    :param n: The number of cutouts to plot, or None to plot all. Default: 100
    :param color: The line color as a pyplot line/marker style. Default: 'k'=black
    :param show: Set this to False to disable showing the plot. Default: True
    """
    if n is None:
        n = cutouts.shape[0]
    # CHANGED
    n = min(n, cutouts.shape[0])
    time_in_us = np.arange(-pre*1000, post*1000, 1e3/fs)

    if show:
        _ = plt.figure(figsize=(12,6))
    
    for i in range(n):
        _ = plt.plot(time_in_us, cutouts[i,]*1e6, color, linewidth=1, alpha=0.3)
        _ = plt.xlabel('Time (%s)' % ureg.ms)
        _ = plt.ylabel('Voltage (%s)' % ureg.uV)
        _ = plt.title('Cutouts')
    plt.savefig("./imgs/Waveforms.png")
    if show:
        plt.show()

