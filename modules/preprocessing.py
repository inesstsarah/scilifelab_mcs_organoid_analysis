'''Module to preprocess the signal using filters, etc'''
import numpy as np
import McsPy
import config
from modules.filters import wavelet_filter, lowpass_filter, bandpass_filter

def preprocess_file(file_path: str, CHANNEL_NMR: int): 
    '''Function to handle, parse, and preprocess a HD5 file'''
    file = file = McsPy.McsData.RawData(file_path)

    # Get the analog signal
    electrode_stream = file.recordings[0].analog_streams[0]
    # Get sampling frequency from file
    fs = (getattr(electrode_stream.channel_infos[0], 'sampling_frequency')) 

    # Check the labels for each index of the stream

    # Check the labels for each index of the stream
    channel_index = 0
    for i in range(0,60):
        temp = electrode_stream.channel_infos[i]
        channel_number = temp.info['Label'][-2:]
        print(f"index: {i}, channel number: {channel_number}")
        if(channel_number!="ef"):
            if(CHANNEL_NMR == int(channel_number)):
                channel_index = i
                break

    print(f"The index chosen is, {channel_index}")

    # Get signal from channel_index from 0 to length of signal
    signal = electrode_stream.get_channel_in_range(channel_index, 0, electrode_stream.channel_data.shape[1])[0]
    return signal, electrode_stream, fs

def preprocess_signal(signal: np.ndarray, fs: int):
    '''Function to preprocess a single signal'''
    
    # Get cutoff from config
    if(config.FILTER_LOWPASS == True):
        lpf_cutoff = config.LOWPASS_CUTOFF
        signal = lowpass_filter(signal, int(fs), lpf_cutoff)
    if(config.FILTER_BANDPASS == True):
        bandpass_cutoffs = (config.BANDPASS_LOW,config.BANDPASS_HIGH)
        signal = bandpass_filter(signal, fs, bandpass_cutoffs[0], bandpass_cutoffs[1], order=3)
    if(config.FILTER_WAVELET == True):
        signal = wavelet_filter(signal, wavelet=config.wavelet_filter["NAME"], level=config.wavelet_filter["LEVEL"], mode=config.wavelet_filter["MODE"], threshold_scale=config.wavelet_filter["TH"])

    filtered_signal = signal
    return filtered_signal

    
    

