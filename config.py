

# Filename for processing
FILENAME = "C:/Users/Ines/Documents/Multi Channel DataManager/2025/SciLifeLab/08.09.25 - Anna Herland/2025-09-08T13-10-30AH-BrainOrganoid-Brainphys-10 min_B-00023.h5"

# Preprocessing
# Lowpass filter cutoff frequency
LOWPASS_CUTOFF = 50

# Highpass filter cutoff frequency
HIGHPASS_CUTOFF = 0.5

# Bandpass filter cutoffs
BANDPASS_LOW = 2
BANDPASS_HIGH = 60 # To filter out electrical line noise 

# Preprocessing steps
FILTER_BANDPASS = True
FILTER_LOWPASS = True

# Choose 1 channel from the file to analyze.
CHANNEL_NMR = 46
# Choose 2 timepoints for the visualizing (but not always necessary)