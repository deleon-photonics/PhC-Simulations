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

#Number of mirror holes to simulate
num_mir_list = np.linspace(0,40,11)
Qvals = []
res_wvls = []

C1 = so.hole_phc(amir       = 172e-9,
                 acav       = 156e-9,
                 wz         = 220e-9,
                 wy         = 430e-9,
                 hx         = 71e-9,
                 hy         = 186e-9,
                 num_cav    = 16,
                 num_mir    = 0,
                 index      = n_GaAs,
                 substrate_index = n_diamond)

for num_mir in num_mir_list:
    C1.num_mir = num_mir
    [Q, res_wvl] = sr.PhC_Q_Simulation(cavity = C1,
                        sim_wvl = wavelength,
                        save_mode_profiles = False)
    Qvals.append(Q)
    res_wvls.append(res_wvl)

plt.plot(num_mir_list, Qvals, marker='o', linestyle='-', color='b', label="Q vs num_mir")
# Add labels and title
plt.xlabel("Num. Mirror Holes")
plt.ylabel("Q-factor")
plt.yscale('log') 
plt.savefig('Q_vs_num_mir.png', format='png', dpi=300)
plt.close()

plt.plot(num_mir_list, np.array(res_wvls)*1e9, marker='o', linestyle='-', color='b', label="Q vs num_mir")
# Add labels and title
plt.xlabel("Num. Mirror Holes")
plt.ylabel("Wavelength [nm]")
plt.savefig('res_wvl_vs_num_mir.png', format='png', dpi=300)
plt.close()

results = {'cavity' : C1,
           'num_mir_list' : num_mir_list,
           'Qvals' : Qvals,
           'res_wvls' : res_wvls}
filename = os.path.join(os.getcwd(), 'C1_Q_scaling')
scipy.io.savemat((filename + '.mat'), {'results': results})
pickle.dump(results, open((filename + '.p'), "wb"))
            