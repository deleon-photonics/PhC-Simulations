from collections import OrderedDict
import numpy as np
import random 
import string


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
        self.sim_time = 1e-12
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
        if self.dimension == '3D':
            fdtd_properties = OrderedDict([("x", self.x),
                           ("y", self.y),
                           ("z", self.z),
                           ("x span", self.xspan),
                           ("y span", self.yspan),
                           ("z span", self.zspan),
                           ("mesh accuracy", self.mesh_accuracy),
                           ("dimension", self.dimension),
                           ("simulation time", self.sim_time),
                           ("use early shutoff", self.early_shutoff),
                           ("x min bc", self.xmin_bc),
                           ("y min bc", self.ymin_bc),
                           ("z min bc", self.zmin_bc),
                           ("x max bc", self.max_bc),
                           ("y max bc", self.max_bc),
                           ("z max bc", self.max_bc)])
        else:
            fdtd_properties = OrderedDict([("x", self.x),
                           ("y", self.y),
                           ("z", self.z),
                           ("x span", self.xspan),
                           ("y span", self.yspan),
                           ("mesh accuracy", self.mesh_accuracy),
                           ("dimension", self.dimension),
                           ("simulation time", self.sim_time),
                           ("use early shutoff", self.early_shutoff),
                           ("x min bc", self.xmin_bc),
                           ("y min bc", self.ymin_bc),
                           ("x max bc", self.max_bc),
                           ("y max bc", self.max_bc)])

        sim.addfdtd(properties=fdtd_properties)
        
class FDE:
    def __init__(self, **kwargs):
        self.x = 0
        self.y = 0
        self.xspan = 1e-6
        self.yspan = 1e-6
        self.max_bc = "PML"
        self.xmin_bc = "PML"
        self.ymin_bc = "PML"
        self.wavelength = 1550e-9
        self.ntrials = 10
        self.bent_wg = 0
        self.bend_radius = 10e-6

        # Update properties with any provided keyword arguments
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def add_to_sim(self, sim):
        fde_properties = OrderedDict([
            ("x", self.x),
            ("y", self.y),
            ("x span", self.xspan),
            ("y span", self.yspan),
            ("x min bc", self.xmin_bc),
            ("y min bc", self.ymin_bc),
            ("x max bc", self.max_bc),
            ("y max bc", self.max_bc),
            ("wavelength", self.wavelength),
            ("number of trial modes", self.ntrials),
            ("bent waveguide", self.bent_wg),
            ("bend radius", self.bend_radius)])

        sim.addfde(properties=fde_properties)    

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
        self.phase = 0
        self.theta = 0
        self.phi = 0

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
            ("wavelength stop", self.wvl_stop),
            ("phase", self.phase),
            ("theta", self.theta),
            ("phi", self.phi)
            ])

        sim.adddipole(properties = dipole_properties)

class port:
    def __init__(self, **kwargs):
        self.name = 'port'
        self.x = 0
        self.y = 0
        self.z = 0
        self.xspan = 1e-6
        self.yspan = 1e-6
        self.zspan = 1e-6
        self.axis = 'x-axis'
        self.direction = 'Forward'
        self.mode = 'Fundamental TE Mode'

        # Update properties with any provided keyword arguments
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def add_to_sim(self, sim):
        port_properties = [
            ("name", self.name),
            ("injection axis", self.axis),
            ("x", self.x), 
            ("y", self.y), 
            ("z", self.z),
            ("x span", self.xspan),
            ("y span", self.yspan),
            ("z span", self.zspan),
            ("direction", self.direction),
            ("mode selection", self.mode)]

        sim.addport()
        for property in port_properties:
            sim.set(property[0], property[1])

    def get_name(self):
        return "FDTD::ports::" + self.name

#Monitors
class Qanalysis:
    def __init__(self, **kwargs):
        self.name = 'Q' + ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(10))
        self.t_start = 0.5e-12
        self.fmin = 0
        self.fmax = 0
        self.x = 20e-9
        self.y = 20e-9
        self.z = 20e-9
        self.xspan = 10e-9
        self.yspan = 10e-9
        self.zspan = 10e-9
        self.nx = 2
        self.ny = 2
        self.nz = 2

        # Update properties with any provided keyword arguments
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def add_to_sim(self, sim):
        Q_properties =  [
            ('name', self.name),
            ("use relative coordinates", 0),
            ("make plots", 0),
            ("x", self.x), 
            ("x span", self.xspan),
            ("y", self.y), 
            ("y span", self.yspan),
            ("z", self.z), 
            ("z span", self.zspan),
            ("nx", self.nx), 
            ("ny", self.ny), 
            ("nz", self.nz),
            ("f min", self.fmin), 
            ("f max", self.fmax),
            ("t start", self.t_start)
        ]

        sim.addobject("Qanalysis")
        for property in Q_properties:
            sim.set(property[0], property[1])

    def get_name(self):
        return self.name
    
class DFT_monitor:
    def __init__(self, **kwargs):
        self.name = 'DFT_' + ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(10))
        self.x = 0
        self.y = 0
        self.z = 0
        self.type = 'Point'
        self.num_freqs = 1
        self.wvl_center = 1550e-9
        self.wvl_span = 0
        self.source_limits = 1
        self.override = 1
        self.apodization = 'none'
        self.apodization_center = 1000e-15
        self.apodization_width = 750e-15
        self.xspan = 1e-6
        self.yspan = 1e-6
        self.zspan = 1e-6

        # Update properties with any provided keyword arguments
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def add_to_sim(self, sim):        
        DFT_props = [
            ('name', self.name),
            ("monitor type", self.type),
            ("use relative coordinates", 0),
            ("x", self.x), 
            ("y", self.y), 
            ("z", self.z), 
            ('override global monitor settings', self.override),
            ('use source limits', self.source_limits),
            ('frequency points', self.num_freqs),
            ('wavelength center', self.wvl_center)
            ]
        if self.apodization != 'none':
            DFT_props.extend([('apodization', self.apodization),
                ('apodization center', self.apodization_center),
                ('apodization time width', self.apodization_width)
            ])
        if self.type == '2D Z-normal':
            DFT_props.extend([('y span', self.yspan),
                              ('x span', self.xspan)])
        elif self.type == '2D Y-normal':
            DFT_props.extend([('z span', self.zspan),
                              ('x span', self.xspan)])
        elif self.type == '2D X-normal':
            DFT_props.extend([('z span', self.zspan),
                              ('y span', self.yspan)])
        DFT_props = OrderedDict(DFT_props)
        sim.addpower(properties = DFT_props)

    def get_name(self):
        return self.name

class time_monitor:
    def __init__(self, **kwargs):
        self.name = 'time_' + ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(10))
        self.x = 0
        self.y = 0
        self.z = 0
        self.type = 'Point'
        self.start_time = 0
        self.min_sampling = 10
        #self.xspan = 1e-6
        #self.yspan = 1e-6
        #self.zspan = 1e-6

        # Update properties with any provided keyword arguments
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def add_to_sim(self, sim):
        time_props = OrderedDict([
            ('name', self.name),
            ("monitor type", self.type),
            ("use relative coordinates", 0),
            ("x", self.x),
            ("y", self.y),
            ("z", self.z),
            ("start time", self.start_time), 
            ("output power", 1),
            ("output Hx", 0), ("output Hy", 0), ("output Hz", 0),
            ("output Ex", 1), ("output Ey", 1), ("output Ez", 1),
            ('min sampling per cycle', self.min_sampling)
            ])
        sim.addtime(properties = time_props)
        

    def get_name(self):
        return self.name

class Mode_Volume_Monitor:
    def __init__(self, **kwargs):
        self.name = 'ModeV' + ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(10))
        self.x = 0
        self.y = 0
        self.z = 0
        self.xspan = 100e-9
        self.yspan = 100e-9
        self.zspan = 100e-9
        self.analysis_wavelength = 955e-9

        # Update properties with any provided keyword arguments
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def add_to_sim(self, sim):
        ModeV_props = [("use relative coordinates", 0),
                   ("x", self.x), ("x span", self.xspan),
                   ("y", self.y), ("y span", self.yspan),
                   ("z", self.z), ("z span", self.zspan),
                   ("calc type", 2)]

        sim.addobject("mode_volume")
        for property in ModeV_props:
            sim.set(property[0], property[1])

        ModeV_field_props = [
            ('override global monitor settings', 1),
            ('use source limits', 0),
            ('frequency points', 1),
            ('wavelength center', self.analysis_wavelength)
            ]
        for property in ModeV_field_props:
            sim.setnamed("mode_volume::field", property[0], property[1])
            sim.setnamed("mode_volume::index", property[0], property[1])


    def get_name(self):
        return self.name
    
#Geometric Objects
###########################
#Hole phc
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

    def use_custom_taper(self):
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
        self.mesh_order = 2

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
                           ("index", self.index),
                           ('name', self.name),
                           ("override mesh order from material database", 1),
                           ("mesh order", self.mesh_order)])

        sim.addcircle(properties=disk_properties)
         
class rectangular_grating:
    def __init__(self, **kwargs):
        self.name = "grating"
        self.x_edge = 0
        self.y = 0
        self.z = 0
        self.period = 1e-6
        self.duty_cycle = 0.5
        self.num_gratings = 10
        self.wy = 1e-6
        self.wz = 1e-6
        self.material = "<Object defined dielectric>"
        self.index = 2.4

        # Update properties with any provided keyword arguments
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def add_to_sim(self, sim):
        xpos = self.x_edge + self.period*(1-self.duty_cycle)
        for i in range(self.num_gratings):
            grating_properties = OrderedDict([
                ("x min", xpos), ("x max", xpos + self.period*self.duty_cycle), 
                ("y", self.y),
                ("z", self.z),
                ("z span", self.wz),
                ("y span", self.wy),
                ("material", self.material),
                ("index", self.index),
                ('name', self.name + str(i))])
            sim.addrect(properties=grating_properties)
            xpos = xpos + self.period

class waveguide:
    def __init__(self, **kwargs):
        self.name = "waveguide"
        self.x = 0
        self.y = 0
        self.z = 0
        self.wx = 1e-6
        self.wy = 1e-6
        self.wz = 1e-6
        self.material = "<Object defined dielectric>"
        self.index = 2.4

        # Update properties with any provided keyword arguments
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def add_to_sim(self, sim):
        waveguide_properties = OrderedDict([("x", self.x),
                           ("y", self.y),
                           ("z", self.z),
                           ("x span", self.wx),
                           ("y span", self.wy),
                           ("z span", self.wz),
                           ("material", self.material),
                           ("index", self.index),
                           ('name', self.name)])

        sim.addrect(properties=waveguide_properties)
         
class polygon:
    def __init__(self, **kwargs):
        self.name = "polygon"
        self.x = 0
        self.y = 0
        self.z = 0
        self.zspan = 1e-6
        self.vertices = [[0, 0], [1, 0], [0.5, 1]]
        self.material = "<Object defined dielectric>"
        self.index = 2.4

        # Update properties with any provided keyword arguments
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def add_to_sim(self, sim):
        taper_props = OrderedDict([("x", self.x),
                           ("y", self.y),
                           ("z", self.z),
                           ("z span", self.zspan),
                           ("material", self.material),
                           ("index", self.index),
                           ('name', self.name)])

        sim.addpoly(properties=taper_props)
        sim.set("vertices", self.vertices)
         