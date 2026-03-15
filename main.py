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
from modules.filters import lowpass_filter, bandpass_filter, wavelet_filter
from modules.preprocessing import preprocess_file, preprocess_signal
from modules.processing import thresholding_voltage
from modules.visualization import plot_processed_signal, plot_spike_raster, plot_mean_waveform, plot_GMM_clusters, plot_autocorrelogram
from modules.analysis import isi_function, pca_dimension_reduction, gmm_clustering, umap_dimension_reduction, kmeans_clustering
import config

# Open file
FILE_PATH = config.FILENAME
# Get specific analog signal using index from config
CHANNEL_NMR = config.CHANNEL_NMR 
signal, electrode_stream, fs = preprocess_file(FILE_PATH, CHANNEL_NMR)
#NOTE: this is just for the purposes of the demo, usually would use whole signal but there is one really big spike
#signal = signal[0:2000000]
# Filtering
#duration = electrode_stream.channel_data.shape[1]/fs.magnitude
duration = len(signal)/fs.magnitude
print(duration)
# Plot original unfiltered signal
plot_signal(signal=signal, fs = fs.magnitude, title=f"Unfiltered Signal of Channel {CHANNEL_NMR}", dur = duration)

filtered_signal = preprocess_signal(signal=signal, fs=fs.magnitude)

# Plot the filtered signal
plot_signal(signal=filtered_signal, fs = fs.magnitude, title=f"Filtered Signal of Channel {CHANNEL_NMR}", dur = duration)
spikes_in_range, cutouts = thresholding_voltage(filtered_signal, fs.magnitude, electrode_stream)
plot_processed_signal(signal=signal, filtered_signal=filtered_signal, fs=fs.magnitude, spikes_in_range = spikes_in_range, dur = duration)

plot_spike_raster(spikes_in_range)

# Analyze  ISI to get the violation percentage
isi_function(spikes_in_range)

# Plot the mean waveform
plot_mean_waveform(cutouts,fs.magnitude)

# Plot autocorrelogram
plot_autocorrelogram(spikes_in_range)

# Do PCA and UMAP analysis with GMM
if(config.PCA_ANALYSIS == True and config.GMM_CLUSTERING == True):
    transformed = pca_dimension_reduction(config.PCA_NUMBER,cutouts)
    labels = gmm_clustering(config.GMM_COMPONENTS, transformed)
    plot_GMM_clusters(cutouts, labels, config.GMM_COMPONENTS,pre=0.0015,post=0.0025, fs = fs.magnitude, title = "PCA and GMM Clustering")
elif(config.UMAP_ANALYSIS == True and config.GMM_CLUSTERING == True):
    transformed = umap_dimension_reduction(cutouts)
    labels = gmm_clustering(config.GMM_COMPONENTS, transformed)
    plot_GMM_clusters(cutouts, labels, config.GMM_COMPONENTS,pre=0.0015,post=0.0025, fs = fs.magnitude, title = "UMAP and GMM Clustering")

if(config.PCA_ANALYSIS == True and config.KMEANS_CLUSTERING == True):
    transformed = pca_dimension_reduction(config.PCA_NUMBER,cutouts)
    labels = kmeans_clustering(config.KMEANS_CLUSTERS, transformed)
    plot_GMM_clusters(cutouts, labels, config.KMEANS_CLUSTERS,pre=0.0015,post=0.0025, fs = fs.magnitude, title = "PCA and KMeans Clustering")

elif(config.UMAP_ANALYSIS == True and config.KMEANS_CLUSTERING == True):
    transformed = umap_dimension_reduction(cutouts)
    labels = kmeans_clustering(config.KMEANS_CLUSTERS, transformed)
    plot_GMM_clusters(cutouts, labels, config.KMEANS_CLUSTERS,pre=0.0015,post=0.0025, fs = fs.magnitude, title = "PCA and KMeans Clustering")








