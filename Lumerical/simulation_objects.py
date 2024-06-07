from collections import OrderedDict
import numpy as np

#Simulation Objects
#FDTD Object
class FDTD:
    def __init__(self, **kwargs):
        self.x = 0
        self.y = 0
        self.z = 0
        self.xspan = 1e-6
        self.yspan = 1e-6
        self.zspan = 1e-6
        self.mesh_accuracy = 2
        self.sim_time = 1e-2
        self.early_shutoff = 1
        self.dimension = "3D"
        self.max_bc = "PML"
        self.xmin_bc = "PML"
        self.ymin_bc = "PML"
        self.zmin_bc = "PML"

        # Update properties with any provided keyword arguments
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def add_to_sim(self, sim):
        fdtd_properties = OrderedDict([("x", self.x),
                           ("y", self.y),
                           ("z", self.z),
                           ("x span", self.xspan),
                           ("y span", self.yspan),
                           ("z span", self.zspan),
                           ("mesh accuracy", self.xspan),
                           ("dimension", self.dimension),
                           ("simulation time", self.sim_time),
                           ("use early shutoff", self.early_shutoff),
                           ("x min bc", self.xmin_bc),
                           ("y min bc", self.ymin_bc),
                           ("z min bc", self.zmin_bc),
                           ("x max bc", self.max_bc),
                           ("y max bc", self.max_bc),
                           ("z max bc", self.max_bc)])

        sim.addfdtd(properties=fdtd_properties)

#Mesh Object
class mesh:
    def __init__(self, **kwargs):
        self.x = 0
        self.y = 0
        self.z = 0
        self.xspan = 1e-6
        self.yspan = 1e-6
        self.zspan = 1e-6
        self.x_resolution = 10e-9
        self.y_resolution = 10e-9
        self.z_resolution = 10e-9

        # Update properties with any provided keyword arguments
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def add_to_sim(self, sim):
        mesh_properties = OrderedDict([("x", self.x),
                           ("y", self.y),
                           ("z", self.z),
                           ("x span", self.xspan),
                           ("y span", self.yspan),
                           ("z span", self.zspan),
                           ("set maximum mesh step", 1),
                           ("override x mesh", 1), 
                           ("override y mesh", 1), 
                           ("override z mesh", 1),
                           ("dx", self.x_resolution), 
                           ("dy", self.y_resolution), 
                           ("dz", self.z_resolution)])

        sim.addmesh(properties = mesh_properties)

#Sources
#Dipole source
class dipole:
    def __init__(self, **kwargs):
        self.x = 0
        self.y = 0
        self.z = 0
        self.dipole_type = "Magnetic dipole"
        self.wvl_start = 1e-6,
        self.wvl_stop = 1e-6

        # Update properties with any provided keyword arguments
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def add_to_sim(self, sim):
        dipole_properties = OrderedDict([
            ("dipole type", self.dipole_type),
            ("x", self.x), 
            ("y", self.y), 
            ("z", self.z),
            ("wavelength start", self.wvl_start),
            ("wavelength stop", self.wvl_stop)
            ])

        sim.adddipole(properties = dipole_properties)

#Monitors
class Qanalysis:
    def __init__(self, **kwargs):
        self.t_start = 0.5e-12
        self.fmin = 0
        self.fmax = 0
        self.x = 20e-9
        self.y = 20e-9
        self.z = 20e-9
        self.xspan = 10e-9
        self.yspan = 10e-9
        self.zspan = 10e-9

        # Update properties with any provided keyword arguments
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def add_to_sim(self, sim):
        Q_properties =  [
            ("use relative coordinates", 0),
            ("make plots", 0),
            ("x", self.x), 
            ("x span", self.xspan),
            ("y", self.y), 
            ("y span", self.yspan),
            ("z", self.z), 
            ("z span", self.zspan),
            ("nx", 2), 
            ("ny", 2), 
            ("nz", 2),
            ("f min", self.fmin), 
            ("f max", self.fmax),
            ("t start", self.t_start)
        ]

        sim.addobject("Qanalysis")
        for property in Q_properties:
            sim.set(property[0], property[1])

#Geometric Objects
###########################
#Nominal hole phc
class hole_phc:
    def __init__(self, **kwargs):
        self.name = "phc"

        self.amir = 100e-9
        self.acav = 100e-9
        self.wz = 100e-9
        self.wy = 100e-9
        self.hx = 100e-9
        self.hy = 100e-9

        self.num_cav = 16
        self.num_mir = 0
        self.taper_exponent = 2

        self.period_list = []
        self.hx_list = []
        self.hy_list = []

        self.hx_error = 0
        self.hy_error = 0
        self.wy_error = 0
        self.wz_error = 0
        self.period_error = 0

        self.custom_taper = []

        self.index = 3.5
        self.substrate_index = 2.4

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def generate_hole_list_nominal(self):
        mirror_holes = self.amir*np.ones(int(self.num_mir/2))

        N = int(self.num_cav/2)
        self.period_list = np.zeros(N)
        a = (self.amir - self.acav)/ ((N-1) ** self.taper_exponent)
        b = self.acav

        for i in range(N):
            self.period_list[i] = a*(i ** self.taper_exponent) + b

        self.period_list = np.append(self.period_list, mirror_holes)
        self.period_list = np.append(np.fliplr(self.period_list), self.period_list)

        self.hx_list     = self.hx*np.ones((self.num_cav + self.num_mir))
        self.hy_list     = self.hy*np.ones((self.num_cav + self.num_mir))
    
    def add_noise(self):
        for i in range(len(self.period_list)):
            self.period_list[i] = np.random.normal(self.period_list[i], self.period_error)
            self.hx_list[i] = np.random.normal(self.hx_list[i], self.hx_error)
            self.hy_list[i] = np.random.normal(self.hy_list[i], self.hy_error)
        self.wy = np.random.normal(self.wy, self.wy_error)
        self.wz = np.random.normal(self.wz, self.wz_error) 

    def set_custom_taper(self):
        mirror_holes = self.amir*np.ones(int(self.num_mir/2))
        self.period_list = np.append(self.custom_taper, mirror_holes)
        self.period_list = np.append(np.fliplr(self.period_list), self.period_list)

    def add_substrate(self, sim):
        substrate_properties = OrderedDict([
            ("x", 0), 
            ("x span", (self.num_cav + self.num_mir + 60)*self.amir + 10e-6),
            ("y", 0), 
            ("y span", 30*self.wy),
            ("z min", -30*self.wz), 
            ("z max", -self.wz/2),
            ("material", "<Object defined dielectric>"), 
            ("index", self.substrate_index),
            ("override mesh order from material database", 1),
            ("mesh order", 3)
        ])
        sim.addrect(properties = substrate_properties)

    def add_to_sim(self, sim):    
        beam_properties = OrderedDict([
            ("x", 0), 
            ("x span", (self.num_cav + self.num_mir + 60)*self.amir + 10e-6),
            ("y", 0), 
            ("y span", self.wy),
            ("z", 0), 
            ("z span", self.wz),
            ("material", "<Object defined dielectric>"), 
            ("index", self.index),
            ("override mesh order from material database", 1),
            ("mesh order", 2)
        ])
        sim.addrect(properties = beam_properties)

        middle_index_left = len(self.period_list)//2 - 1
        middle_index_right = len(self.period_list)//2

        xpos_r = -self.period_list[middle_index_right]/2
        xpos_l = self.period_list[middle_index_left]/2
        for i in range(len(self.period_list)//2):
            xpos_r += self.period_list[middle_index_right + i]
            xpos_l -= self.period_list[middle_index_left - i]

            right_hole = OrderedDict([
                ("name", "right_hole_" + str(i)),
                ("make ellipsoid", 1),
                ("x", xpos_r), 
                ("z", 0), 
                ("z span", self.wz),
                ("radius", self.hx_list[middle_index_right + i]/2), 
                ("radius 2", self.hy_list[middle_index_right + i]/2),
                ("material", "<Object defined dielectric>"), 
                ("index", 1),
                ("override mesh order from material database", 1),
                ("mesh order", 1)
            ])

            left_hole = OrderedDict([
                ("name", "left_hole_" + str(i)),
                ("make ellipsoid", 1),
                ("x", xpos_l), 
                ("z", 0), 
                ("z span", self.wz),
                ("radius", self.hx_list[middle_index_left - i]/2), 
                ("radius 2", self.hy_list[middle_index_left - i]/2),
                ("material", "<Object defined dielectric>"), 
                ("index", 1),
                ("override mesh order from material database", 1),
                ("mesh order", 1)
            ])

            sim.addcircle(properties = right_hole)
            sim.addcircle(properties = left_hole)


class microdisk:
    def __init__(self, **kwargs):
        self.name = "disk"
        self.x = 0
        self.y = 0
        self.z = 0
        self.thickness = 1e-6
        self.radius = 1e-6
        self.material = "<Object defined dielectric>"
        self.index = 2.4

        # Update properties with any provided keyword arguments
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def add_to_sim(self, sim):
        disk_properties = OrderedDict([("x", self.x),
                           ("y", self.y),
                           ("z", self.z),
                           ("z span", self.thickness),
                           ("radius", self.radius),
                           ("material", self.material),
                           ("index", self.index)])

        sim.addcircle(properties=disk_properties)
         