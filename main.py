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


# import bombcell module
import bombcell as bc

import sys
from pathlib import Path
from pprint import pprint 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import bombcell as bc

# Import from modules
from modules.filters import lowpass_filter
import config


FILE_PATH = config.FILENAME
file = file = McsPy.McsData.RawData(FILE_PATH)
electrode_stream = file.recordings[0].analog_streams[0]

# Get sampling frequency from file
fs = (getattr(electrode_stream.channel_infos[0], 'sampling_frequency')) 

# Filtering
# Do processing on all the signals within electrode stream

# Delete unresponsive/uninformative import config
import modules.filters
import modules.utils

# Choose pre-processing method
