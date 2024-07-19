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

R = np.array([10000, 20000, 30000, 40000, 50000])
T = np.array([250, 500, 750, 1250, 1500, 1750, 2000])

combined_data = []

base_dir = r'Z:\deleonlab\data\Diamond Brillouin Laser\Simulations\2024-07-09-R-T-Sweep-Overlap'  # Use raw string literal

# Iterate over all combinations of R and T
for r in R:
    for t in T:
        folder_name = f'r_{r}nm_t_{t}nm_U_5000nm'
        folder_path = os.path.join(base_dir, folder_name)
        
        print(f'Checking folder: {folder_path}')  # Debug statement
        
        if os.path.isdir(folder_path):
            file_path = os.path.join(folder_path, 'output.p')
            if os.path.isfile(file_path):
                print(f'Loading file: {file_path}')  # Debug statement
                with open(file_path, 'rb') as file:
                    try:
                        data = pickle.load(file)
                        delta_f = data['delta_f']
                        mode_overlap = data['mode_overlap']
                        
                        # Append each entry in delta_f and mode_overlap with corresponding r and t
                        for df, mo in zip(delta_f, mode_overlap):
                            combined_data.append([r, t, df, mo])
                    except Exception as e:
                        print(e)
            else:
                print(f'File not found: {file_path}')  # Debug statement
        else:
            print(f'Folder not found: {folder_path}')  # Debug statement

# Convert the combined data to a NumPy array
combined_array = np.array(combined_data)

# Save to CSV
csv_path = os.path.join(base_dir, 'combined_data_5umUndercut.csv')
df = pd.DataFrame(combined_array, columns=['r', 't', 'delta_f', 'mode_overlap'])
df.to_csv(csv_path, index=False)

# Save to .mat
mat_path = os.path.join(base_dir, 'combined_data_5umUndercut.mat')
savemat(mat_path, {'combined_data': combined_array})

print(f'Data saved to {csv_path} and {mat_path}')

#import simulation_objects as so
#import simulation_routines as sr

#from mpl_toolkits.mplot3d import Axes3D

""" 

with open('2024-07-03_r_t_sweep_UndercutSweep.p', 'rb') as f:
    data = pickle.load(f)
wavelength = 1550e-9

Q = data['Q_factors']
res_wvls = data['resonance_wavelengths']
decay_lengths = data['decay_lengths']
disk_radii = data['radii']
disk_t = data['thicknesses']

repeated_disk_radii = []
repeated_disk_t = []

flat_Q = [item for sublist in Q for item in sublist]
flat_wvl = [item for sublist in res_wvls for item in sublist]
flat_decay = [item for sublist in decay_lengths for item in sublist]

flat_Q = np.array(flat_Q)
flat_wvl = np.array(flat_wvl)
flat_decay = np.array(flat_decay)


for i in range(len(Q)):
    repeated_disk_radii.extend(disk_radii[i]*np.ones(len(Q[i])))
    repeated_disk_t.extend(disk_t[i]*np.ones(len(Q[i])))

repeated_disk_radii = np.array(repeated_disk_radii)
repeated_disk_t = np.array(repeated_disk_t)

unique_t = np.unique(repeated_disk_t)

plt.figure()

for i in range(len(unique_t)):
    plt.scatter(repeated_disk_radii[repeated_disk_t == unique_t[i]]*1e6, flat_Q[repeated_disk_t == unique_t[i]], label=f't = {unique_t[i]*1e9:.0f}')

plt.xlabel('Radius (um)')
plt.ylabel('Q Factor')
plt.yscale('log')
plt.legend()
plt.title('Q Factor vs Radius')
plt.grid(True)
plt.savefig('Q_vs_Radius.png')
plt.clf()  # Clear the figure for the next plot


# Plot decay lengths vs radius
for i in range(len(unique_t)):
    plt.scatter(repeated_disk_radii[repeated_disk_t == unique_t[i]]*1e6, flat_decay[repeated_disk_t == unique_t[i]]*1e6, label=f't = {unique_t[i]*1e9:.0f}')

plt.xlabel('Radius (um)')
plt.ylabel('Decay Length (nm)')
plt.title('Decay Length vs Radius')
plt.grid(True)
plt.legend()

plt.savefig('Decay_Length_vs_Radius.png')
plt.clf()  # Clear the figure


for i in range(len(unique_t)):
    plt.scatter(flat_decay[repeated_disk_t == unique_t[i]]*1e6, flat_Q[repeated_disk_t == unique_t[i]],label=f't = {unique_t[i]*1e9:.0f}')

plt.xlabel('Decay Length (nm)')
plt.ylabel('Q')
plt.yscale('log')
plt.title('Q vs Decay Length')
plt.grid(True)
plt.legend()

plt.savefig('Q_vs_DL.png')
plt.clf()  # Clear the figure """