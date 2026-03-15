

# Filename for processing
FILENAME = "C:/Users/Ines/Documents/Multi Channel DataManager/2025/SciLifeLab/08.09.25 - Anna Herland/2025-09-08T13-10-30AH-BrainOrganoid-Brainphys-10 min_B-00023.h5"

# Choose 1 channel from the file to analyze.
CHANNEL_NMR = 50

# Choose which preprocessing steps to do 
FILTER_BANDPASS = False
FILTER_LOWPASS = True
FILTER_WAVELET = False 

# Preprocessing
# Lowpass filter cutoff frequency
LOWPASS_CUTOFF = 50

# Highpass filter cutoff frequency
HIGHPASS_CUTOFF = 0.5

# Bandpass filter cutoffs
BANDPASS_LOW = 2
BANDPASS_HIGH = 60 # To filter out electrical line noise 

# Wavelet filter params
wavelet_filter = {'NAME':"db4",'LEVEL':None,'MODE':"soft",'TH':1.0}

# Choose thresholding method

# Choose 2 timepoints for the visualizing (but not always necessary)