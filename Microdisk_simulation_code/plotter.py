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

import simulation_objects as so
import simulation_routines as sr

from mpl_toolkits.mplot3d import Axes3D



with open('2024-06-21_r_sweep.p', 'rb') as f:
    data = pickle.load(f)

wavelength = 850e-9
disk_radii = np.linspace(1.25*wavelength,10*wavelength, 36)

Q = data['Q_factors']
res_wvls = data['resonance_wavelengths']
decay_lengths = data['decay_lengths']

repeated_disk_radii = []
flat_Q = [item for sublist in Q for item in sublist]
flat_wvl = [item for sublist in res_wvls for item in sublist]
flat_decay = [item for sublist in decay_lengths for item in sublist]

flat_Q = np.array(flat_Q)
flat_wvl = np.array(flat_wvl)
flat_decay = np.array(flat_decay)

for i in range(len(Q)):
    repeated_disk_radii.extend(disk_radii[i]*np.ones(len(Q[i])))

repeated_disk_radii = np.array(repeated_disk_radii)
plt.figure()

plt.scatter(repeated_disk_radii*1e6, flat_Q)#, flat_Q, color='g')

plt.xlabel('Radius (um)')
plt.ylabel('Q Factor')
plt.yscale('log')
plt.title('Q Factor vs Radius')
plt.grid(True)
plt.savefig('Q_vs_Radius.png')
plt.clf()  # Clear the figure for the next plot

# Plot resonance wavelengths vs radius
plt.scatter(repeated_disk_radii*1e6, flat_wvl*1e9)  # Convert wavelengths to nm for better visualization
plt.xlabel('Radius (um)')
plt.ylabel('Resonance Wavelength (nm)')
plt.title('Resonance Wavelength vs Radius')
plt.grid(True)
plt.savefig('Resonance_Wavelength_vs_Radius.png')
plt.clf()  # Clear the figure for the next plot

# Plot decay lengths vs radius
plt.scatter(repeated_disk_radii*1e6, flat_decay*1e9)
plt.xlabel('Radius (um)')
plt.ylabel('Decay Length (nm)')
plt.title('Decay Length vs Radius')
plt.grid(True)
plt.savefig('Decay_Length_vs_Radius.png')
plt.clf()  # Clear the figure


plt.scatter(flat_decay*1e9, flat_Q)
plt.xlabel('Decay Length (nm)')
plt.ylabel('Q')
plt.yscale('log')
plt.title('Q vs Decay Length')
plt.grid(True)
plt.savefig('Q_vs_DL.png')
plt.clf()  # Clear the figure

Qmax_ind = np.argmax(flat_Q)
print(repeated_disk_radii[Qmax_ind]*1e9)

print("Plots have been saved successfully.")