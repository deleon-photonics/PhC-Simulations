from collections import OrderedDict
import lumapi as lp
import numpy as np
import pickle
import math
import os
import copy
from datetime import date
import matplotlib.pyplot as plt
import random
import scipy.optimize
import pandas as pd
import time
from scipy.stats import norm
import scipy
from scipy.io import savemat

combined_data = []

base_dir = r'Z:\deleonlab\data\Diamond Brillouin Laser\Simulations\2024-07-09-R-T-Sweep-Overlap'  # Use raw string literal

r = 10000
t = 2000

folder_name = f'r_{r}nm_t_{t}nm_U_5000nm'
folder_path = os.path.join(base_dir, folder_name)

if os.path.isdir(folder_path):
    file_path = os.path.join(folder_path, 'output.p')
    if os.path.isfile(file_path):
        print(f'Loading file: {file_path}')  # Debug statement
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            delta_f = list(data['delta_f'])
            mode_overlap = list(data['mode_overlap'])
            res_wvl = data['resonance_wavelengths']

            if len(res_wvl) == len(delta_f) + 1:
                delta_f.insert(0, 0)  # Insert a zero at the beginning
                mode_overlap.insert(0, 0)  # Insert a zero at the beginning
            
            # Append each entry in delta_f, mode_overlap, and res_wvl with corresponding r and t
            for df, mo, res in zip(delta_f, mode_overlap, res_wvl):
                combined_data.append([r, t, df/1e9, mo, res*1e9])
# Convert the combined data to a NumPy array
combined_array = np.array(combined_data)

# Save to CSV
csv_path = os.path.join(folder_path, 'combined_data.csv')
df = pd.DataFrame(combined_array, columns=['r', 't', 'FSR (GHz)', 'mode_overlap', 'res_wvl (nm)'])
df.to_csv(csv_path, index=False)

# Save to .mat
mat_path = os.path.join(folder_path, 'combined_data.mat')
savemat(mat_path, {'combined_data': combined_array})

print(f'Data saved to {csv_path} and {mat_path}')
