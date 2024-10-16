#Example simulation of Q-scaling of hole-based PhC
#Cavity "C1", described in arXiv:2409.16883v1

import simulation_objects as so
import simulation_routines as sr
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
import pickle


#Material Constants
n_GaAs = 3.55
n_diamond = 2.4
wavelength = 955e-9

#Number of noisy simulations to perform
num_iter = 30
Q_noisy = np.zeros(num_iter)
res_wvls_noisy = np.zeros(num_iter)

C1 = so.hole_phc(amir       = 172e-9,
                 acav       = 156e-9,
                 wz         = 220e-9,
                 wy         = 430e-9,
                 hx         = 71e-9,
                 hy         = 186e-9,
                 num_cav    = 16,
                 num_mir    = 12,
                 index      = n_GaAs,
                 substrate_index = n_diamond)

#Compute the nominal Q and wavelength
[Q_nominal, res_wvl_nominal] = sr.PhC_Q_Simulation(cavity = C1,
                        sim_wvl = wavelength,
                        save_mode_profiles = False,
                        cavity_name = 'C1_nominal',
                        min_boundary_conditions=["PML", "anti-symmetric", "PML"])

#Error values for noisy simulations
C1.noisy_cavity = True
C1.hx_error = 0.02*C1.hx
C1.hy_error = 0.02*C1.hy
C1.wy_error = 0.01*C1.wy
C1.period_error = 0.75e-9
C1.wz_error = 0

for i in range(num_iter):
    [Q_noisy[i], res_wvls_noisy[i]] = sr.PhC_Q_Simulation(cavity = C1,
                        sim_wvl = wavelength,
                        save_mode_profiles = False,
                        min_boundary_conditions=["PML", "anti-symmetric", "PML"])

plt.scatter(res_wvls_noisy * 1e9, Q_noisy, marker='o', color='b')
plt.scatter(res_wvl_nominal * 1e9, Q_nominal, marker='*', color='r')
# Add labels and title
plt.xlabel("Wavelength [nm]")
plt.ylabel("Q-factor")
plt.yscale('log') 
plt.savefig('Q_vs_wvl_noisy.png', format='png', dpi=300)
plt.close()


results = {'cavity'             : C1,
           'Q_nominal'          : Q_nominal,
           'Q_noisy'            : Q_noisy,
           'res_wvl_nominal'    : res_wvl_nominal,
           'res_wvls_noisy'     : res_wvls_noisy}
filename = os.path.join(os.getcwd(), 'C1_noise_analysis')
scipy.io.savemat((filename + '.mat'), {'results': results})
pickle.dump(results, open((filename + '.p'), "wb"))
            