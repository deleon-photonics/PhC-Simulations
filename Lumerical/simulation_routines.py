import simulation_objects as so
import lumapi as lp
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os
import pickle
import scipy.io


#Microdisk resonances
def microdisk_resonances(disk_radius, disk_thickness, material_index, wavelength):
    output_folder = "r_" + str(int(disk_radius*1e6)) + "um_t_" + str(int(disk_thickness*1e9)) + "nm"
    os.makedirs(output_folder, exist_ok=True)

    sim = lp.FDTD(hide=True)

    disk = so.microdisk(thickness = disk_thickness, radius = disk_radius, index = material_index)
    disk.add_to_sim(sim)
    
    fdtd = so.FDTD(xspan = disk_radius*2 + 1.5e-6, 
                   yspan = disk_radius*2 + 1.5e-6, 
                   zspan = 2*wavelength, 
                   dimension = '3D', 
                   sim_time = 2e-12,
                   xmin_bc = 'symmetric',
                   ymin_bc = 'anti-symmetric',
                   zmin_bc = 'symmetric')
    fdtd.add_to_sim(sim)

    theta = 45
    for xi in [-1,0,1]:
        for yi in [-1,0,1]:
            x = np.cos(theta*np.pi/180)*disk_radius - wavelength/4 + xi*wavelength/8
            y = np.sin(theta*np.pi/180)*disk_radius - wavelength/4 + yi*wavelength/8
            dipole = so.dipole(x=x, y=y, z = 0, wvl_start = wavelength - 25e-9, wvl_stop = wavelength + 25e-9, dipole_type = 'Magnetic Dipole', theta=0, phi=0)
            dipole.add_to_sim(sim)

    theta = 25
    x = np.cos(theta*np.pi/180)*disk_radius - wavelength/8
    y = np.sin(theta*np.pi/180)*disk_radius - wavelength/8
    DFT_monitor = so.DFT_monitor(x=x, y=y-wavelength/8, num_freqs = 5000, wvl_center = wavelength, apodization = 'full')
    DFT_monitor.add_to_sim(sim)
    time_monitor = so.time_monitor(x=x-wavelength/8, y = y, start_time = 0, num_freqs = 5000)
    time_monitor.add_to_sim(sim)
    
    #mesh = so.mesh(xspan = 2*disk_radius, yspan = 2*disk_radius)
    #mesh.add_to_sim(sim)

    if os.path.isfile('gui.fps'):
        os.remove('gui.fsp')
    sim.save('gui.fsp')
    sim.run()
    sim.runanalysis()

    spectrum = sim.getresult(DFT_monitor.get_name(), 'E')
    E = spectrum['E']
    E = E[0,0,0,:,:]
    wvl = spectrum['lambda']

    Emag = np.sqrt(np.abs(np.sum(E * np.conjugate(E), axis = 1)))

    peaks,_ = find_peaks(Emag, height=0.01*np.max(Emag), distance=50)
    res_wvls = wvl[peaks]

    plt.figure(figsize=(8, 6))
    plt.plot(wvl, Emag, label='|E| vs wvl')
    plt.plot(wvl[peaks], Emag[peaks], 'o', label='Peaks')
    plt.xlabel('X axis')
    plt.ylabel('|E|')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, 'E_vs_wvl.png'))
    plt.close()

    """ sim.switchtolayout()
    DFT_monitor_list = [None] * len(res_wvls)
    for i in range(len(res_wvls)):
        res = res_wvls[i]
        DFT_monitor_list[i] = so.DFT_monitor(type='2D Z-normal',
                                     source_limits = 0,
                                     x=0, 
                                     y=0, 
                                     xspan = disk_radius*2 + 1.5e-6, 
                                     yspan = disk_radius*2 + 1.5e-6,
                                     num_freqs = 1, 
                                     wvl_center = res, 
                                     apodization = 'full',
                                     apodization_center = 500e-15,
                                     apodization_width = 100e-15)
        DFT_monitor_list[i].add_to_sim(sim)
    sim.run()

    #Example for monitor 0
    decay_lengths = np.zeros_like(res_wvls)
    i = 0
    for field_profile in DFT_monitor_list:
        field_profile_data = sim.getresult(field_profile.get_name(), 'E')
        E_ = field_profile_data['E']
        E = E_[:,:,0,0,:]
        x = field_profile_data['x']
        y = field_profile_data['y']
        E_sq = np.abs((np.sum(E * np.conjugate(E), axis = 2)))

        x0 = np.argmin(np.abs(x))
        y0 = np.argmin(np.abs(y))

        x_pos = x[x0:]
        y_pos = y[y0:]

        sliced_E_sq = E_sq[x0:, y0:]

        max = 0
        for j in range(0, len(x_pos)):
            for k in range(0, len(y_pos)):
                if sliced_E_sq[j, k] > max:
                    max =  sliced_E_sq[j, k]
                    max_index_sliced = (j, k)
        sliced_E_sq = sliced_E_sq / max
        max_x = x_pos[max_index_sliced[1]]
        max_y = y_pos[max_index_sliced[0]]

        tangent_slope = -max_x / max_y
        perpendicular_slope = -1 / tangent_slope
        x1 = x_pos[0]
        y1 = max_y + perpendicular_slope *(x1 - max_x)
        x2 = x_pos[-1]
        y2 = max_y + (x2-max_x)*perpendicular_slope

        plt.figure(figsize=(8, 6))
        plt.imshow(sliced_E_sq, extent=[np.min(x_pos), np.max(x_pos), np.min(y_pos), np.max(y_pos)], origin='lower', cmap='viridis', aspect='auto')
        plt.plot([x1, x2], [y1, y2], 'r--')  # Plotting dashed line using indices
        plt.xlim(x_pos.min(), x_pos.max())
        plt.ylim(y_pos.min(), y_pos.max())
        plt.colorbar(label='Magnitude')  # Add a color bar to show the magnitude scale
        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        plt.title('|E|^2')
        plt.savefig(os.path.join(output_folder, 'Esq_' + str(int(res_wvls[i]*1e9)) + 'nm_.png'))
        plt.close()
        

        E_1d = []
        y_1d = []
        r_1d = []
        yvals = max_y + (x_pos - max_x)*perpendicular_slope
        for j in range(len(x_pos)):
            yind = np.argmin(np.abs(y_pos - yvals[j]))
            E_1d.append(sliced_E_sq[yind, j])
            y_1d.append(y[yind])
            r_1d.append(np.sqrt(x_pos[j]**2 + y_pos[yind]**2))
        
        threshold = 0.01
        for l, value in enumerate(E_1d):
            if value > threshold:
                index = l
                break
        
        plt.figure(figsize=(8, 6))
        plt.plot(r_1d, E_1d, label='|E|^2 vs radial length')
        plt.axvline(x=disk_radius, color='r', linestyle='--')
        plt.xlabel('r (um)')
        plt.ylabel('|E|^2')
        plt.title(f"99pcnt field decay length: {(disk_radius - r_1d[index])*1e6} um")
        plt.xlim((0, disk_radius*3))
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_folder, 'E_1D_' + str(int(res_wvls[i]*1e9)) + 'nm_.png'))
        plt.close()

        decay_lengths[i] = disk_radius - r_1d[index]

        i = i + 1
    results = {'resonance_wavelengths': res_wvls,
               'decay_lengths': decay_lengths}
    

    with open(os.path.join(output_folder, 'output.p'), 'wb') as f:
        pickle.dump(results, f)
    scipy.io.savemat(os.path.join(output_folder, 'output.mat'), results)
    return results, res_wvls, decay_lengths """
    

        

        

