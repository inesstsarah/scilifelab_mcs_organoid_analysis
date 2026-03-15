

# Filename for processing
FILENAME = "C:/Users/Ines/Documents/Multi Channel DataManager/2025/SciLifeLab/08.09.25 - Anna Herland/2025-09-08T13-10-30AH-BrainOrganoid-Brainphys-10 min_B-00023.h5"

# Choose 1 channel from the file to analyze.
CHANNEL_NMR = 46

# Choose which preprocessing steps to do 
FILTER_BANDPASS = True
FILTER_LOWPASS = False
FILTER_WAVELET = False 

# Preprocessing
# Lowpass filter cutoff frequency
LOWPASS_CUTOFF = 50

# Highpass filter cutoff frequency
HIGHPASS_CUTOFF = 0.5

# Bandpass filter cutoffs
BANDPASS_LOW = 300
BANDPASS_HIGH = 3000 # To get spikes 

# Wavelet filter params
wavelet_filter = {'NAME':"db4",'LEVEL':None,'MODE':"soft",'TH':1.0}

# Dimensional reduction params
PCA_ANALYSIS = True
UMAP_ANALYSIS = False

PCA_NUMBER = 2

# Clustering params
KMEANS_CLUSTERING = True
GMM_CLUSTERING = False
GMM_COMPONENTS = 3
KMEANS_CLUSTERS = 3