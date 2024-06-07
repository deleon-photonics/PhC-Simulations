#2023-08-05
#Code for running Lumerical FDTD Simulations of Hybrid PhCs

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

import simulation_objects as so




#Simulation Functions
def Q_V_Simulation(cavity_parameters, high_resolution = 1, 
                   mesh_resolution = [10e-9, 10e-9, 10e-9]):
    amir        = cavity_parameters['amir']
    wz          = cavity_parameters['wz']
    wy          = cavity_parameters['wy']
    num_cav     = cavity_parameters['num_cav']
    num_mir     = cavity_parameters['num_mir']
    num_tap     = cavity_parameters['num_tap']

    #FDTD Object
    mesh_accuracy           = 2
    fdtd_sim_time           = 1e-12
    fdtd_early_shutoff      = 0
    fdtd_dimension          = "3D"
    fdtd_yz_size            = (math.ceil(wy/amir) + 8)*amir
    fdtd_x                  = 0
    fdtd_xspan              = (num_cav + num_mir + 2*num_tap + 60)*amir
    fdtd_y                  = 0
    fdtd_yspan              = fdtd_yz_size
    fdtd_z                  = 0
    fdtd_zspan              = fdtd_yz_size
    fdtd_max_bc             = "PML"
    fdtd_xmin_bc            = "symmetric"
    fdtd_ymin_bc            = "anti-symmetric"

    #Mesh object
    mesh_name               = "mesh"
    mesh_x                  = 0
    mesh_xspan              = (num_cav + num_mir + 2*num_tap + 2)*amir#fdtd_xspan + 2e-6
    mesh_y                  = 0
    mesh_yspan              = 1.5*wy #fdtd_yspan + 2e-6
    mesh_z                  = 0
    mesh_zspan              = 1.5*wz#fdtd_zspan + 2e-6

    #Dipole sources
    dipole1_name            = "dipole 1"
    dipole1_type            = "Magnetic dipole"
    dipole1_x               = 10e-9
    dipole1_y               = 20e-9
    dipole1_z               = 40e-9
    dipole1_wvl_start       = 0.95*wavelength
    dipole1_wvl_stop        = 1.05*wavelength

    dipole2_name            = "dipole 2"
    dipole2_type            = "Magnetic dipole"
    dipole2_x               = 20e-9
    dipole2_y               = 30e-9
    dipole2_z               = 10e-9
    dipole2_wvl_start       = 0.97*wavelength
    dipole2_wvl_stop        = 1.03*wavelength

    #Q-analysis object
    Q_t_start               = 0.5e-12
    Qanalysis_lambda_min    = wavelength - 46e-9
    Qanalysis_lambda_max    = wavelength + 54e-9
    Qanalysis_f_min         = (3e8)/Qanalysis_lambda_max
    Qanalysis_f_max         = (3e8)/Qanalysis_lambda_min
    Q_use_rel_coords        = 0
    Q_x                     = 20e-9
    Q_y                     = 20e-9
    Q_z                     = wz/6
    Q_xspan                 = 10e-9
    Q_yspan                 = 10e-9
    Q_zspan                 = wz/3
    Q_nx                    = 2
    Q_ny                    = 2
    Q_nz                    = 2

    #Mode Volume Object
    ModeV_use_rel_coords    = 0
    ModeV_x                 = 0
    ModeV_y                 = 0
    ModeV_z                 = 0
    ModeV_x_span            = (num_cav + num_mir + 2*num_tap + 2)*amir
    ModeV_y_span            = fdtd_yz_size - 3*amir
    ModeV_z_span            = fdtd_yz_size - 3*amir
    ModeV_calc_type         = 2

    #Power Monitors
    xmax_monitor_name       = "x max transmission monitor"
    xmax_monitor_type       = "2D X-normal"
    xmax_monitor_x          = fdtd_xspan/2 - 500e-9
    xmax_monitor_y          = 0
    xmax_monitor_yspan      = fdtd_yspan
    xmax_monitor_z          = 0
    xmax_monitor_zspan      = fdtd_zspan

    ymax_monitor_name       = "y max transmission monitor"
    ymax_monitor_type       = "2D Y-normal"
    ymax_monitor_x          = 0
    ymax_monitor_xspan      = fdtd_xspan
    ymax_monitor_y          = fdtd_yspan/2 - 300e-9
    ymax_monitor_z          = 0
    ymax_monitor_zspan      = fdtd_zspan

    zmin_monitor_name       = "z min transmission monitor"
    zmin_monitor_type       = "2D Z-normal"
    zmin_monitor_x          = 0
    zmin_monitor_xspan      = fdtd_xspan
    zmin_monitor_y          = 0
    zmin_monitor_yspan      = fdtd_yspan
    zmin_monitor_z          = -fdtd_zspan/2 + 300e-9

    zmax_monitor_name       = "z max tansmission monitor"
    zmax_monitor_z          = -1*zmin_monitor_z

    #Define dictionaries for the simulation objects
    fdtd_props = OrderedDict([
        ("dimension", fdtd_dimension), ("simulation time", fdtd_sim_time), 
        ("use early shutoff", fdtd_early_shutoff), ("mesh accuracy", mesh_accuracy),
        ("x", fdtd_x), ("x span", fdtd_xspan), 
        ("y", fdtd_y), ("y span", fdtd_yspan),
        ("z", fdtd_z), ("z span", fdtd_zspan),
        ("x max bc", fdtd_max_bc), ("y max bc", fdtd_max_bc),
        ("z min bc", fdtd_max_bc), ("z max bc", fdtd_max_bc),
        ("x min bc", fdtd_xmin_bc), ("y min bc", fdtd_ymin_bc)
    ])

    dipole1_props = OrderedDict([
        ("name", dipole1_name),
        ("dipole type", dipole1_type),
        ("x", dipole1_x), ("y", dipole1_y), ("z", dipole1_z),
        ("wavelength start", dipole1_wvl_start),
        ("wavelength stop", dipole1_wvl_stop)
    ])

    dipole2_props = OrderedDict([
        ("name", dipole2_name),
        ("dipole type", dipole2_type),
        ("x", dipole2_x), ("y", dipole2_y), ("z", dipole2_z),
        ("wavelength start", dipole2_wvl_start),
        ("wavelength stop", dipole2_wvl_stop)
    ])

    mesh_props = OrderedDict([
        ("name", mesh_name),
        ("x", mesh_x), ("x span", mesh_xspan),
        ("y", mesh_y), ("y span", mesh_yspan),
        ("z", mesh_z), ("z span", mesh_zspan),
        ("set maximum mesh step", 1),
        ("override x mesh", 1), ("override y mesh", 1), ("override z mesh", 1),
        ("dx", mesh_resolution[0]), ("dy", mesh_resolution[1]), ("dz", mesh_resolution[2])
    ])

    Q_props = [
        ("use relative coordinates", Q_use_rel_coords),
        ("make plots", 0),
        ("x", Q_x), ("x span", Q_xspan),
        ("y", Q_y), ("y span", Q_yspan),
        ("z", Q_z), ("z span", Q_zspan),
        ("nx", Q_nx), ("ny", Q_ny), ("nz", Q_nz),
        ("f min", Qanalysis_f_min), ("f max", Qanalysis_f_max),
        ("t start", Q_t_start)
    ]

    ModeV_props = [("use relative coordinates", ModeV_use_rel_coords),
                   ("x", ModeV_x), ("x span", ModeV_x_span),
                   ("y", ModeV_y), ("y span", ModeV_y_span),
                   ("z", ModeV_z), ("z span", ModeV_z_span),
                   ("calc type", ModeV_calc_type)]
    
    xmax_monitor_props = OrderedDict([
        ("name", xmax_monitor_name), ("monitor type", xmax_monitor_type),
        ("use relative coordinates", 0),
        ("x", xmax_monitor_x), 
        ("y", xmax_monitor_y), ("y span", xmax_monitor_yspan),
        ("z", xmax_monitor_z), ("z span", xmax_monitor_zspan),
        ("start time", Q_t_start), ("output power", 1),
        ("output Hx", 0), ("output Hy", 0), ("output Hz", 0),
        ("output Ex", 0), ("output Ey", 0), ("output Ez", 0)
    ])

    ymax_monitor_props = OrderedDict([
        ("name", ymax_monitor_name), ("monitor type", ymax_monitor_type),
        ("use relative coordinates", 0),
        ("x", ymax_monitor_x), ("x span", ymax_monitor_xspan),
        ("y", ymax_monitor_y),
        ("z", ymax_monitor_z), ("z span", ymax_monitor_zspan),
        ("start time", Q_t_start), ("output power", 1),
        ("output Hx", 0), ("output Hy", 0), ("output Hz", 0),
        ("output Ex", 0), ("output Ey", 0), ("output Ez", 0)
    ])

    zmin_monitor_props = OrderedDict([
        ("name", zmin_monitor_name), ("monitor type", zmin_monitor_type),
        ("use relative coordinates", 0),
        ("x", zmin_monitor_x), ("x span", zmin_monitor_xspan),
        ("y", zmin_monitor_y), ("y span", zmin_monitor_yspan),
        ("z", zmin_monitor_z), 
        ("start time", Q_t_start), ("output power", 1),
        ("output Hx", 0), ("output Hy", 0), ("output Hz", 0),
        ("output Ex", 0), ("output Ey", 0), ("output Ez", 0)
    ])

    zmax_monitor_props = OrderedDict([
        ("name", zmax_monitor_name), ("monitor type", zmin_monitor_type),
        ("use relative coordinates", 0),
        ("x", zmin_monitor_x), ("x span", zmin_monitor_xspan),
        ("y", zmin_monitor_y), ("y span", zmin_monitor_yspan),
        ("z", zmax_monitor_z), 
        ("start time", Q_t_start), ("output power", 1),
        ("output Hx", 0), ("output Hy", 0), ("output Hz", 0),
        ("output Ex", 0), ("output Ey", 0), ("output Ez", 0)
    ])

    #Add the objects to the simulation
    sim.addfdtd(properties = fdtd_props)
    sim.setnamed("FDTD", "x span", (num_cav + num_mir + 2*num_tap + 8)*amir)
    sim.adddipole(properties = dipole1_props)
    sim.adddipole(properties = dipole2_props)

    if high_resolution == 1:
        sim.addmesh(properties = mesh_props)

    sim.addobject("Qanalysis")
    for property in Q_props:
        sim.set(property[0], property[1])
    sim.addanalysisresult("f0")

    if os.path.isfile('gui.fps'):
        os.remove('gui.fsp')
    sim.save('gui.fsp')

    sim.run()
    sim.runanalysis()

    Qcal            = sim.getresult("Qanalysis", "Q")
    maxQ            = np.max(Qcal['Q'])
    ind_maxQ        = np.argmax(Qcal['Q'])
    lambda_maxQ     = Qcal['lambda'][ind_maxQ]
    f_maxQ          = Qcal['f'][ind_maxQ]

    sim.switchtolayout()
    sim.setnamed("FDTD", "x span", (num_cav + num_mir + 2*num_tap + 60)*amir)

    sim.addobject("mode_volume")
    for property in ModeV_props:
        sim.set(property[0], property[1])

    ModeV_field_props = [
        ('override global monitor settings', 1),
        ('use source limits', 0),
        ('frequency points', 1),
        ('wavelength center', lambda_maxQ)
        ]
    for property in ModeV_field_props:
        sim.setnamed("mode_volume::field", property[0], property[1])
        sim.setnamed("mode_volume::index", property[0], property[1])

    sim.addtime(properties = xmax_monitor_props)
    sim.addtime(properties = ymax_monitor_props)
    sim.addtime(properties = zmin_monitor_props)
    sim.addtime(properties = zmax_monitor_props)

    sim.run()
    sim.runanalysis()

    ModeV           = sim.getresult("mode_volume", "Volume")
    ModeV_norm      = np.array(ModeV['V']) / ((lambda_maxQ / material_index)**3)
    ModeV           = ModeV['V']

    [Ex, Ey, Ez]    = get_energy(xmax_monitor=xmax_monitor_name,
                                 ymax_monitor=ymax_monitor_name,
                                 zmin_monitor=zmin_monitor_name,
                                 zmax_monitor=zmax_monitor_name)
    
    [Qx, Qy, Qz]    = get_Q_vector(Qtotal=maxQ,
                                   freq=f_maxQ,
                                   Ex=Ex, Ey=Ey, Ez=Ez,
                                   sample_time=(fdtd_sim_time - Q_t_start))
    
    return [maxQ, lambda_maxQ, ModeV, ModeV_norm, Ex,  Ey, Ez, Qx, Qy, Qz]

def Q_Simulation(cavity_parameters,
                 ysize = 0,
                 zsize = 0, 
                 high_resolution = 1, 
                 mesh_resolution = [10e-9, 10e-9, 10e-9],
                 mesh_setting = 2):
    amir        = cavity_parameters['amir']
    wz          = cavity_parameters['wz']
    wy          = cavity_parameters['wy']
    num_cav     = cavity_parameters['num_cav']
    num_mir     = cavity_parameters['num_mir']
    num_tap     = cavity_parameters['num_tap']

    #FDTD Object
    mesh_accuracy           = mesh_setting
    fdtd_sim_time           = 1e-12
    fdtd_early_shutoff      = 0
    fdtd_dimension          = "3D"
    fdtd_yz_size            = (math.ceil(wy/amir) + 8)*amir
    fdtd_x                  = 0
    fdtd_xspan              = (num_cav + num_mir + 2*num_tap + 10)*amir
    fdtd_y                  = 0
    fdtd_z                  = 0
    fdtd_max_bc             = "PML"
    fdtd_xmin_bc            = "symmetric"
    fdtd_ymin_bc            = "anti-symmetric"
    if ysize == 0:
        fdtd_yspan          = fdtd_yz_size
    else:
        fdtd_yspan          = ysize
    
    if zsize == 0:
        fdtd_zspan          = fdtd_yz_size
    else:
        fdtd_zspan          = zsize

    #Mesh object
    mesh_name               = "mesh"
    mesh_x                  = 0
    mesh_xspan              = fdtd_xspan + 2e-6 #
    mesh_y                  = 0
    mesh_yspan              = fdtd_yspan + 2e-6 
    mesh_z                  = 0
    mesh_zspan              = fdtd_zspan + 2e-6

    #Dipole sources
    dipole1_name            = "dipole 1"
    dipole1_type            = "Magnetic dipole"
    dipole1_x               = 10e-9
    dipole1_y               = 20e-9
    dipole1_z               = 40e-9
    dipole1_wvl_start       = 0.95*wavelength
    dipole1_wvl_stop        = 1.05*wavelength

    dipole2_name            = "dipole 2"
    dipole2_type            = "Magnetic dipole"
    dipole2_x               = 20e-9
    dipole2_y               = 30e-9
    dipole2_z               = 10e-9
    dipole2_wvl_start       = 0.97*wavelength
    dipole2_wvl_stop        = 1.03*wavelength

    #Q-analysis object
    Q_t_start               = 0.5e-12
    Qanalysis_lambda_min    = wavelength - 46e-9
    Qanalysis_lambda_max    = wavelength + 54e-9
    Qanalysis_f_min         = (3e8)/Qanalysis_lambda_max
    Qanalysis_f_max         = (3e8)/Qanalysis_lambda_min
    Q_use_rel_coords        = 0
    Q_x                     = 20e-9
    Q_y                     = 20e-9
    Q_z                     = wz/6
    Q_xspan                 = 10e-9
    Q_yspan                 = 10e-9
    Q_zspan                 = wz/3
    Q_nx                    = 2
    Q_ny                    = 2
    Q_nz                    = 2

    #Define dictionaries for the simulation objects
    fdtd_props = OrderedDict([
        ("dimension", fdtd_dimension), ("simulation time", fdtd_sim_time), 
        ("use early shutoff", fdtd_early_shutoff), ("mesh accuracy", mesh_accuracy),
        ("x", fdtd_x), ("x span", fdtd_xspan), 
        ("y", fdtd_y), ("y span", fdtd_yspan),
        ("z", fdtd_z), ("z span", fdtd_zspan),
        ("x max bc", fdtd_max_bc), ("y max bc", fdtd_max_bc),
        ("z min bc", fdtd_max_bc), ("z max bc", fdtd_max_bc),
        ("x min bc", fdtd_xmin_bc), ("y min bc", fdtd_ymin_bc)
    ])

    dipole1_props = OrderedDict([
        ("name", dipole1_name),
        ("dipole type", dipole1_type),
        ("x", dipole1_x), ("y", dipole1_y), ("z", dipole1_z),
        ("wavelength start", dipole1_wvl_start),
        ("wavelength stop", dipole1_wvl_stop)
    ])

    dipole2_props = OrderedDict([
        ("name", dipole2_name),
        ("dipole type", dipole2_type),
        ("x", dipole2_x), ("y", dipole2_y), ("z", dipole2_z),
        ("wavelength start", dipole2_wvl_start),
        ("wavelength stop", dipole2_wvl_stop)
    ])

    mesh_props = OrderedDict([
        ("name", mesh_name),
        ("x", mesh_x), ("x span", mesh_xspan),
        ("y", mesh_y), ("y span", mesh_yspan),
        ("z", mesh_z), ("z span", mesh_zspan),
        ("set maximum mesh step", 1),
        ("override x mesh", 1), ("override y mesh", 1), ("override z mesh", 1),
        ("dx", mesh_resolution[0]), ("dy", mesh_resolution[1]), ("dz", mesh_resolution[2])
    ])

    Q_props = [
        ("use relative coordinates", Q_use_rel_coords),
        ("make plots", 0),
        ("x", Q_x), ("x span", Q_xspan),
        ("y", Q_y), ("y span", Q_yspan),
        ("z", Q_z), ("z span", Q_zspan),
        ("nx", Q_nx), ("ny", Q_ny), ("nz", Q_nz),
        ("f min", Qanalysis_f_min), ("f max", Qanalysis_f_max),
        ("t start", Q_t_start)
    ]

    #Add the objects to the simulation
    sim.addfdtd(properties = fdtd_props)
    sim.setnamed("FDTD", "x span", (num_cav + num_mir + 2*num_tap + 8)*amir)
    sim.adddipole(properties = dipole1_props)
    sim.adddipole(properties = dipole2_props)

    if high_resolution == 1:
        sim.addmesh(properties = mesh_props)

    sim.addobject("Qanalysis")
    for property in Q_props:
        sim.set(property[0], property[1])
    sim.addanalysisresult("f0")

    if os.path.isfile('gui.fps'):
        os.remove('gui.fsp')
    sim.save('gui.fsp')

    sim.run()
    sim.runanalysis()

    Qcal            = sim.getresult("Qanalysis", "Q")
    maxQ            = np.max(Qcal['Q'])
    ind_maxQ        = np.argmax(Qcal['Q'])
    lambda_maxQ     = Qcal['lambda'][ind_maxQ]
    f_maxQ          = Qcal['f'][ind_maxQ]
    
    return [maxQ, lambda_maxQ]

def Q_Simulation_Asymmetric(cavity_parameters, high_resolution = 1, 
                 mesh_resolution = [10e-9, 10e-9, 10e-9],
                 mesh_setting = 2):
    amir        = cavity_parameters['amir']
    wz          = cavity_parameters['wz']
    wy          = cavity_parameters['wy']
    num_cav     = cavity_parameters['num_cav']
    num_mir     = cavity_parameters['num_mir']
    num_tap     = cavity_parameters['num_tap']

    #FDTD Object
    mesh_accuracy           = mesh_setting
    fdtd_sim_time           = 2e-12
    fdtd_early_shutoff      = 0
    fdtd_dimension          = "3D"
    fdtd_yz_size            = (math.ceil(wy/amir) + 8)*amir
    fdtd_x                  = 0
    fdtd_xspan              = (num_cav + num_mir + 2*num_tap + 10)*amir
    fdtd_y                  = 0
    fdtd_yspan              = fdtd_yz_size
    fdtd_z                  = 0
    fdtd_zspan              = fdtd_yz_size
    fdtd_max_bc             = "PML"
    fdtd_xmin_bc            = "PML"
    fdtd_ymin_bc            = "anti-symmetric"

    #Mesh object
    mesh_name               = "mesh"
    mesh_x                  = 0
    mesh_xspan              = (num_cav + num_mir + 2*num_tap + 2)*amir #fdtd_xspan + 2e-6 #
    mesh_y                  = 0
    mesh_yspan              = 1.5*wy #fdtd_yspan + 2e-6 
    mesh_z                  = 0
    mesh_zspan              = 1.5*wz #fdtd_zspan + 2e-6

    #Dipole sources
    dipole1_name            = "dipole 1"
    dipole1_type            = "Magnetic dipole"
    dipole1_x               = 10e-9
    dipole1_y               = 20e-9
    dipole1_z               = 40e-9
    dipole1_wvl_start       = 0.95*wavelength
    dipole1_wvl_stop        = 1.05*wavelength

    dipole2_name            = "dipole 2"
    dipole2_type            = "Magnetic dipole"
    dipole2_x               = 20e-9
    dipole2_y               = 30e-9
    dipole2_z               = 10e-9
    dipole2_wvl_start       = 0.97*wavelength
    dipole2_wvl_stop        = 1.03*wavelength

    #Q-analysis object
    Q_t_start               = 0.5e-12
    Qanalysis_lambda_min    = wavelength - 46e-9
    Qanalysis_lambda_max    = wavelength + 54e-9
    Qanalysis_f_min         = (3e8)/Qanalysis_lambda_max
    Qanalysis_f_max         = (3e8)/Qanalysis_lambda_min
    Q_use_rel_coords        = 0
    Q_x                     = 20e-9
    Q_y                     = 20e-9
    Q_z                     = wz/6
    Q_xspan                 = 10e-9
    Q_yspan                 = 10e-9
    Q_zspan                 = wz/3
    Q_nx                    = 2
    Q_ny                    = 2
    Q_nz                    = 2

    #Mode Volume Object
    ModeV_use_rel_coords    = 0
    ModeV_x                 = 0
    ModeV_y                 = 0
    ModeV_z                 = 0
    ModeV_x_span            = (num_cav + num_mir + 2*num_tap + 2)*amir
    ModeV_y_span            = fdtd_yz_size - 3*amir
    ModeV_z_span            = fdtd_yz_size - 3*amir
    ModeV_calc_type         = 2

    xy_monitor_name         = "XY DFT monitor"
    xy_monitor_type       = "2D Z-normal"
    xy_monitor_x          = 0
    xy_monitor_xspan      = fdtd_xspan
    xy_monitor_y          = 0
    xy_monitor_yspan      = fdtd_yspan
    xy_monitor_z          = 0

    #Define dictionaries for the simulation objects
    fdtd_props = OrderedDict([
        ("dimension", fdtd_dimension), ("simulation time", fdtd_sim_time), 
        ("use early shutoff", fdtd_early_shutoff), ("mesh accuracy", mesh_accuracy),
        ("x", fdtd_x), ("x span", fdtd_xspan), 
        ("y", fdtd_y), ("y span", fdtd_yspan),
        ("z", fdtd_z), ("z span", fdtd_zspan),
        ("x max bc", fdtd_max_bc), ("y max bc", fdtd_max_bc),
        ("z min bc", fdtd_max_bc), ("z max bc", fdtd_max_bc),
        ("x min bc", fdtd_xmin_bc), ("y min bc", fdtd_ymin_bc)
    ])

    dipole1_props = OrderedDict([
        ("name", dipole1_name),
        ("dipole type", dipole1_type),
        ("x", dipole1_x), ("y", dipole1_y), ("z", dipole1_z),
        ("wavelength start", dipole1_wvl_start),
        ("wavelength stop", dipole1_wvl_stop)
    ])

    dipole2_props = OrderedDict([
        ("name", dipole2_name),
        ("dipole type", dipole2_type),
        ("x", dipole2_x), ("y", dipole2_y), ("z", dipole2_z),
        ("wavelength start", dipole2_wvl_start),
        ("wavelength stop", dipole2_wvl_stop)
    ])

    mesh_props = OrderedDict([
        ("name", mesh_name),
        ("x", mesh_x), ("x span", mesh_xspan),
        ("y", mesh_y), ("y span", mesh_yspan),
        ("z", mesh_z), ("z span", mesh_zspan),
        ("set maximum mesh step", 1),
        ("override x mesh", 1), ("override y mesh", 1), ("override z mesh", 1),
        ("dx", mesh_resolution[0]), ("dy", mesh_resolution[1]), ("dz", mesh_resolution[2])
    ])

    Q_props = [
        ("use relative coordinates", Q_use_rel_coords),
        ("make plots", 0),
        ("x", Q_x), ("x span", Q_xspan),
        ("y", Q_y), ("y span", Q_yspan),
        ("z", Q_z), ("z span", Q_zspan),
        ("nx", Q_nx), ("ny", Q_ny), ("nz", Q_nz),
        ("f min", Qanalysis_f_min), ("f max", Qanalysis_f_max),
        ("t start", Q_t_start)
    ]
    ModeV_props = [("use relative coordinates", ModeV_use_rel_coords),
                   ("x", ModeV_x), ("x span", ModeV_x_span),
                   ("y", ModeV_y), ("y span", ModeV_y_span),
                   ("z", ModeV_z), ("z span", ModeV_z_span),
                   ("calc type", ModeV_calc_type)]
    
    #Add the objects to the simulation
    sim.addfdtd(properties = fdtd_props)
    sim.setnamed("FDTD", "x span", (num_cav + num_mir + 2*num_tap + 8)*amir)
    sim.adddipole(properties = dipole1_props)
    sim.adddipole(properties = dipole2_props)

    if high_resolution == 1:
        sim.addmesh(properties = mesh_props)

    sim.addobject("Qanalysis")
    for property in Q_props:
        sim.set(property[0], property[1])
    sim.addanalysisresult("f0")

    if os.path.isfile('gui.fps'):
        os.remove('gui.fsp')
    sim.save('gui.fsp')

    sim.run()
    sim.runanalysis()

    Qcal            = sim.getresult("Qanalysis", "Q")
    maxQ            = np.max(Qcal['Q'])
    ind_maxQ        = np.argmax(Qcal['Q'])
    lambda_maxQ     = Qcal['lambda'][ind_maxQ]
    f_maxQ          = Qcal['f'][ind_maxQ]
    
    
    return [maxQ, lambda_maxQ]#, Ex,  Ey, Ez, Qx, Qy, Qz]


def Q_Simulation_Asymmetric_kspace(cavity_parameters, high_resolution = 1, 
                 mesh_resolution = [10e-9, 10e-9, 10e-9],
                 mesh_setting = 2):
    amir        = cavity_parameters['amir']
    wz          = cavity_parameters['wz']
    wy          = cavity_parameters['wy']
    num_cav     = cavity_parameters['num_cav']
    num_mir     = cavity_parameters['num_mir']
    num_tap     = cavity_parameters['num_tap']

    #FDTD Object
    mesh_accuracy           = mesh_setting
    fdtd_sim_time           = 2e-12
    fdtd_early_shutoff      = 0
    fdtd_dimension          = "3D"
    fdtd_yz_size            = (math.ceil(wy/amir) + 8)*amir
    fdtd_x                  = 0
    fdtd_xspan              = (num_cav + num_mir + 2*num_tap + 10)*amir
    fdtd_y                  = 0
    fdtd_yspan              = fdtd_yz_size
    fdtd_z                  = 0
    fdtd_zspan              = fdtd_yz_size
    fdtd_max_bc             = "PML"
    fdtd_xmin_bc            = "PML"
    fdtd_ymin_bc            = "anti-symmetric"

    #Mesh object
    mesh_name               = "mesh"
    mesh_x                  = 0
    mesh_xspan              = (num_cav + num_mir + 2*num_tap + 2)*amir #fdtd_xspan + 2e-6 #
    mesh_y                  = 0
    mesh_yspan              = 1.5*wy #fdtd_yspan + 2e-6 
    mesh_z                  = 0
    mesh_zspan              = 1.5*wz #fdtd_zspan + 2e-6

    #Dipole sources
    dipole1_name            = "dipole 1"
    dipole1_type            = "Magnetic dipole"
    dipole1_x               = 10e-9
    dipole1_y               = 20e-9
    dipole1_z               = 40e-9
    dipole1_wvl_start       = 0.95*wavelength
    dipole1_wvl_stop        = 1.05*wavelength

    dipole2_name            = "dipole 2"
    dipole2_type            = "Magnetic dipole"
    dipole2_x               = 20e-9
    dipole2_y               = 30e-9
    dipole2_z               = 10e-9
    dipole2_wvl_start       = 0.97*wavelength
    dipole2_wvl_stop        = 1.03*wavelength

    #Q-analysis object
    Q_t_start               = 0.5e-12
    Qanalysis_lambda_min    = wavelength - 46e-9
    Qanalysis_lambda_max    = wavelength + 54e-9
    Qanalysis_f_min         = (3e8)/Qanalysis_lambda_max
    Qanalysis_f_max         = (3e8)/Qanalysis_lambda_min
    Q_use_rel_coords        = 0
    Q_x                     = 20e-9
    Q_y                     = 20e-9
    Q_z                     = wz/6
    Q_xspan                 = 10e-9
    Q_yspan                 = 10e-9
    Q_zspan                 = wz/3
    Q_nx                    = 2
    Q_ny                    = 2
    Q_nz                    = 2

    #Mode Volume Object
    ModeV_use_rel_coords    = 0
    ModeV_x                 = 0
    ModeV_y                 = 0
    ModeV_z                 = 0
    ModeV_x_span            = (num_cav + num_mir + 2*num_tap + 2)*amir
    ModeV_y_span            = fdtd_yz_size - 3*amir
    ModeV_z_span            = fdtd_yz_size - 3*amir
    ModeV_calc_type         = 2

    xy_monitor_name         = "XY DFT monitor"
    xy_monitor_type       = "2D Z-normal"
    xy_monitor_x          = 0
    xy_monitor_xspan      = fdtd_xspan
    xy_monitor_y          = 0
    xy_monitor_yspan      = fdtd_yspan
    xy_monitor_z          = 0

    #Define dictionaries for the simulation objects
    fdtd_props = OrderedDict([
        ("dimension", fdtd_dimension), ("simulation time", fdtd_sim_time), 
        ("use early shutoff", fdtd_early_shutoff), ("mesh accuracy", mesh_accuracy),
        ("x", fdtd_x), ("x span", fdtd_xspan), 
        ("y", fdtd_y), ("y span", fdtd_yspan),
        ("z", fdtd_z), ("z span", fdtd_zspan),
        ("x max bc", fdtd_max_bc), ("y max bc", fdtd_max_bc),
        ("z min bc", fdtd_max_bc), ("z max bc", fdtd_max_bc),
        ("x min bc", fdtd_xmin_bc), ("y min bc", fdtd_ymin_bc)
    ])

    dipole1_props = OrderedDict([
        ("name", dipole1_name),
        ("dipole type", dipole1_type),
        ("x", dipole1_x), ("y", dipole1_y), ("z", dipole1_z),
        ("wavelength start", dipole1_wvl_start),
        ("wavelength stop", dipole1_wvl_stop)
    ])

    dipole2_props = OrderedDict([
        ("name", dipole2_name),
        ("dipole type", dipole2_type),
        ("x", dipole2_x), ("y", dipole2_y), ("z", dipole2_z),
        ("wavelength start", dipole2_wvl_start),
        ("wavelength stop", dipole2_wvl_stop)
    ])

    mesh_props = OrderedDict([
        ("name", mesh_name),
        ("x", mesh_x), ("x span", mesh_xspan),
        ("y", mesh_y), ("y span", mesh_yspan),
        ("z", mesh_z), ("z span", mesh_zspan),
        ("set maximum mesh step", 1),
        ("override x mesh", 1), ("override y mesh", 1), ("override z mesh", 1),
        ("dx", mesh_resolution[0]), ("dy", mesh_resolution[1]), ("dz", mesh_resolution[2])
    ])

    Q_props = [
        ("use relative coordinates", Q_use_rel_coords),
        ("make plots", 0),
        ("x", Q_x), ("x span", Q_xspan),
        ("y", Q_y), ("y span", Q_yspan),
        ("z", Q_z), ("z span", Q_zspan),
        ("nx", Q_nx), ("ny", Q_ny), ("nz", Q_nz),
        ("f min", Qanalysis_f_min), ("f max", Qanalysis_f_max),
        ("t start", Q_t_start)
    ]
    ModeV_props = [("use relative coordinates", ModeV_use_rel_coords),
                   ("x", ModeV_x), ("x span", ModeV_x_span),
                   ("y", ModeV_y), ("y span", ModeV_y_span),
                   ("z", ModeV_z), ("z span", ModeV_z_span),
                   ("calc type", ModeV_calc_type)]
    
    #Add the objects to the simulation
    sim.addfdtd(properties = fdtd_props)
    sim.setnamed("FDTD", "x span", (num_cav + num_mir + 2*num_tap + 8)*amir)
    sim.adddipole(properties = dipole1_props)
    sim.adddipole(properties = dipole2_props)

    if high_resolution == 1:
        sim.addmesh(properties = mesh_props)

    sim.addobject("Qanalysis")
    for property in Q_props:
        sim.set(property[0], property[1])
    sim.addanalysisresult("f0")

    if os.path.isfile('gui.fps'):
        os.remove('gui.fsp')
    sim.save('gui.fsp')

    sim.run()
    sim.runanalysis()

    Qcal            = sim.getresult("Qanalysis", "Q")
    maxQ            = np.max(Qcal['Q'])
    ind_maxQ        = np.argmax(Qcal['Q'])
    lambda_maxQ     = Qcal['lambda'][ind_maxQ]
    f_maxQ          = Qcal['f'][ind_maxQ]
    
    sim.switchtolayout()
    #sim.setnamed("FDTD", "x span", (num_cav + num_mir + 2*num_tap + 60)*amir)

    sim.addobject("mode_volume")
    for property in ModeV_props:
        sim.set(property[0], property[1])

    ModeV_field_props = [
        ('override global monitor settings', 1),
        ('use source limits', 0),
        ('frequency points', 1),
        ('wavelength center', lambda_maxQ)
        ]
    for property in ModeV_field_props:
        sim.setnamed("mode_volume::field", property[0], property[1])
        sim.setnamed("mode_volume::index", property[0], property[1])

    xy_monitor_props = OrderedDict([
        ("name", xy_monitor_name), ("monitor type", xy_monitor_type),
        ("use relative coordinates", 0),
        ("x", xy_monitor_x), ("x span", xy_monitor_xspan),
        ("y", xy_monitor_y), ("y span", xy_monitor_yspan),
        ("z", xy_monitor_z), 
        ('override global monitor settings', 1),
        ('use source limits', 0),
        ('frequency points', 1),
        ('wavelength center', lambda_maxQ)
    ])
    sim.addpower(properties = xy_monitor_props)

    sim.run()
    sim.runanalysis()

    ModeV           = sim.getresult("mode_volume", "Volume")
    ModeV_norm      = np.array(ModeV['V']) / ((lambda_maxQ / material_index)**3)
    ModeV           = ModeV['V']
    
    Exy     = sim.getresult(xy_monitor_name, "E")
    Exy_x   = sim.getdata(xy_monitor_name, "x")
    Exy_y   = sim.getdata(xy_monitor_name, "y")
    Ey      = sim.getdata(xy_monitor_name, "Ey")

    FieldProfile = {'Monitor_Data'  : Exy,
                    'x'             : Exy_x,
                    'y'             : Exy_y,
                    'Ey'            : Ey,
                    'Cavity'        : cavity_parameters,
                    'Q'             : maxQ,
                    'wvl'           : lambda_maxQ,
                    'ModeVol'       : ModeV,
                    'ModeVolNorm'   : ModeV_norm}
    filename = f'FieldProfile_{str(iter_fname)}.mat'
    scipy.io.savemat(filename, {'FieldProfile': FieldProfile})
    
    return [maxQ, lambda_maxQ, ModeV, ModeV_norm]#, Ex,  Ey, Ez, Qx, Qy, Qz]


#Sweep functions 
def Ncav_Nmir_Sweep(cavity_parameters, num_cav_hole_list, num_mir_hole_list):
    Qtotal      = np.zeros((len(num_cav_hole_list), len(num_mir_hole_list)))
    res_wvl     = np.zeros((len(num_cav_hole_list), len(num_mir_hole_list)))


    for i in range(len(num_cav_hole_list)):
        cavity_parameters['num_cav'] = num_cav_hole_list[i]
        for j in range(len(num_mir_hole_list)):
            cavity_parameters['num_mir'] = num_mir_hole_list[j]

            sim.switchtolayout()
            sim.deleteall()

            add_hole_phc(cavity_parameters=cavity_parameters)
            add_substrate(cavity_parameters)

            [Qtotal[i,j], res_wvl[i,j]] = Q_Simulation(cavity_parameters=cavity_parameters,
                                                       high_resolution=1,
                                                       mesh_resolution=[10e-9, 16e-9, 16e-9])
            
            results = {
                'Qtotal'        : Qtotal,
                'res_wvl'       : res_wvl,
                'num_cav'       : num_cav_hole_list,
                'num_mir'       : num_mir_hole_list,
                'cavity'        : cavity_parameters
            }
            print(results)
            pickle.dump(results, open(str(run_date) + str(cavity_number) +  "_Qscaling.p", "wb"))



run_date = date.today()

wavelength = 1550e-9
material_index = 2.3682
substrate_index = 1
sim = lp.FDTD(hide=False)

disk = so.microdisk(thickness = 1e-6, radius = 10e-6, index = material_index)
disk.add_to_sim(sim)
fdtd = so.FDTD()
fdtd.add_to_sim(sim)
