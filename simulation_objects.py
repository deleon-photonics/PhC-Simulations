from collections import OrderedDict

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


#Geometric Objects
###########################
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
         