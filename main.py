# Read MCS data
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

import os
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# MCS PyData tools
import McsPy
import McsPy.McsData
from McsPy import ureg, Q_

# VISUALIZATION TOOLS
import matplotlib.pyplot as plt

# SUPRESS WARNINGS
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
from pprint import pprint 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import modules.filters
import modules.utils
from modules.visualization import plot_signal
# Import from modules
from modules.filters import lowpass_filter, bandpass_filter
import config

# Open file
FILE_PATH = config.FILENAME
file = file = McsPy.McsData.RawData(FILE_PATH)

# Get the analog signal
electrode_stream = file.recordings[0].analog_streams[0]
# Get sampling frequency from file
fs = (getattr(electrode_stream.channel_infos[0], 'sampling_frequency')) 


# Get specific analog signal using index from config
CHANNEL_NMR = config.CHANNEL_NMR 

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

# Filtering
duration = electrode_stream.channel_data.shape[1]/fs.magnitude
print(duration)
# Plot original signal
plot_signal(signal=signal, fs = fs.magnitude, title=f"Unfiltered Signal of Channel {CHANNEL_NMR}", dur = duration)

# Get cutoff from config
if(config.FILTER_LOWPASS == True):
    lpf_cutoff = config.LOWPASS_CUTOFF
    signal = lowpass_filter(signal, int(fs.magnitude), lpf_cutoff)
if(config.FILTER_BANDPASS == True):
    bandpass_cutoffs = (config.BANDPASS_LOW,config.BANDPASS_HIGH)
    signal = bandpass_filter(signal, fs.magnitude, bandpass_cutoffs[0], bandpass_cutoffs[1], order=3)

filtered_signal = signal



plot_signal(signal=filtered_signal, fs = fs.magnitude, title=f"Filtered Signal of Channel {CHANNEL_NMR}", dur = duration)




