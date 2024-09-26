import simulation_objects as so
import lumapi as lp
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.signal import find_peaks
import pickle
import scipy.io
import math
from scipy.optimize import curve_fit
from scipy.constants import c
import random
import string
import pandas as pd

def PhC_Q_Simulation(cavity = None,
                    sim_wvl = 955e-9, 
                    min_boundary_conditions = ["symmetric", "anti-symmetric", "PML"],
                    max_boundary_condition = "PML",
                    output_folder =  'Q_simulation', 
                    dimension = "3D",
                    mesh_accuracy = 2,
                    sim_time = 2e12,
                    use_fine_mesh = True,
                    mesh_resolutions = [10e-9, 20e-9, 20e-9],
                    save_mode_profiles = True,
                    cavity_name = 'cavity' + ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(10))):
    
    os.makedirs(output_folder, exist_ok=True)

    sim = lp.FDTD(hide=True)

    #Add the cavity
    cavity.add_to_sim(sim)

    #Add FDTD Object
    fdtd = so.FDTD(x = 0, y = 0, z = 0,
                    xspan = (cavity.num_cav + cavity.num_mir + 10)*cavity.amir,
                    yspan = (math.ceil(cavity.wy/cavity.amir) + 8)*cavity.amir,
                    zspan = (math.ceil(cavity.wy/cavity.amir) + 8)*cavity.amir,
                    mesh_accuracy = mesh_accuracy,
                    dimension = dimension,
                    early_shutoff = 0,
                    max_BC = max_boundary_condition,
                    xmin_bc = min_boundary_conditions[0],
                    ymin_bc = min_boundary_conditions[1],
                    zmin_bc = min_boundary_conditions[2],
                    sim_time = sim_time)
    fdtd.add_to_sim(sim)

    #Add dipole sources
    dipole_1 = so.dipole(type = "Magnetic dipole",
                         x = 10e-9, y = 20e-9, z = 40e-9,
                         wvl_start = 0.95*sim_wvl,
                         wvl_stop = 1.05*sim_wvl)
    dipole_1.add_to_sim(sim)

    dipole_2 = so.dipole(type = "Magnetic dipole",
                         x = 20e-9, y = 30e-9, z = 10e-9,
                         wvl_start = 0.97*sim_wvl,
                         wvl_stop = 1.03*sim_wvl)
    dipole_2.add_to_sim(sim)

    #Q-analysis object
    Q_analysis = so.Qanalysis(t_start = 500e-12,
                              fmin = 3e8/(sim_wvl + 50e-9),
                              fmax = 3e8/(sim_wvl - 50e-9),
                              x = 20e-9, y = 20e-9, z = cavity.wz/6,
                              xspan = 10e-9, yspan = 10e-9, zspan = cavity.wz/3,
                              nx = 2, ny = 2, nz = 2)
    Q_analysis.add_to_sim(sim)

    if use_fine_mesh:
        mesh = so.mesh(x = 0, y = 0, z = 0,
                       xspan = (cavity.num_cav + cavity.num_mir + 2*cavity.num_tap + 10)*cavity.amir,
                       yspan = 1.5*cavity.wy,
                       zspan = 1.5*cavity.wz,
                       x_resolution = mesh_resolutions[0],
                       y_resolution = mesh_resolutions[1],
                       z_resolution = mesh_resolutions[2])
        mesh.add_to_sim(sim)

    try:
        sim.run()
        sim.runanalysis()
        Qcal            = sim.getresult(Q_analysis.get_name(), "Q")
        maxQ            = np.max(Qcal['Q'])
        ind_maxQ        = np.argmax(Qcal['Q'])
        lambda_maxQ     = Qcal['lambda'][ind_maxQ]
        f_maxQ          = Qcal['f'][ind_maxQ]
    
        #Mode volume and mode-profile
        if save_mode_profiles:
            sim.switchtolayout()
            ModeV_Monitor = so.Mode_Volume_Monitor(x = 0, y = 0, z = 0,
                                                xspan = (cavity.num_cav + cavity.num_mir + 2)*cavity.amir,
                                                yspan = (math.ceil(cavity.wy/cavity.amir) + 8)*cavity.amir - 3*cavity.amir,
                                                zspan = (math.ceil(cavity.wy/cavity.amir) + 8)*cavity.amir - 3*cavity.amir,
                                                analysis_wavelength = lambda_maxQ)
            ModeV_Monitor.add_to_sim(sim)

            xy_monitor = so.DFT_monitor(type='2D Z-normal',
                                        source_limits = 0,
                                        x=0, y=0, z = 0, 
                                        xspan = (cavity.num_cav + cavity.num_mir + 10)*cavity.amir, 
                                        yspan = (math.ceil(cavity.wy/cavity.amir) + 8)*cavity.amir,
                                        override = 1, num_freqs = 1, wvl_center = lambda_maxQ)
            xy_monitor.add_to_sim(sim)

            sim.run()
            sim.runanalysis()

            ModeV           = sim.getresult(ModeV_Monitor.get_name(), "Volume")
            ModeV_norm      = np.array(ModeV['V']) / ((lambda_maxQ / cavity.index)**3)
            ModeV           = ModeV['V']
            
            Exy     = sim.getresult(xy_monitor.get_name(), "E")
            Exy_x   = sim.getdata(xy_monitor.get_name(), "x")
            Exy_y   = sim.getdata(xy_monitor.get_name(), "y")
            Ey      = sim.getdata(xy_monitor.get_name(), "Ey")

            FieldProfile = {'Monitor_Data'  : Exy,
                            'x'             : Exy_x,
                            'y'             : Exy_y,
                            'Ey'            : Ey,
                            'Cavity'        : vars(cavity),
                            'Q'             : maxQ,
                            'wvl'           : lambda_maxQ,
                            'ModeVol'       : ModeV,
                            'ModeVolNorm'   : ModeV_norm}
            filename = f'FieldProfile_{str(cavity_name)}'
            scipy.io.savemat((filename + '.mat'), {'FieldProfile': FieldProfile})
            pickle.dump(FieldProfile, open((filename + '.p'), "wb"))
            
            return [maxQ, lambda_maxQ, ModeV, ModeV_norm]
        
        else:
            return [maxQ, lambda_maxQ]
        
    except Exception as e:
        print(f"No Resonance Found: {e}")
        return -1           
            
