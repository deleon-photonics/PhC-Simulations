import simulation_objects as so
import lumapi as lp
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os
import pickle
import scipy.io
from scipy.optimize import curve_fit
from scipy.constants import c


# Fit each peak with a Lorentzian or Gaussian function
def lorentzian(x, x0, gamma, A):
    return A * gamma**2 / ((x - x0)**2 + gamma**2)

def fit_Q(freqs, Esq):
    peaks,_ = find_peaks(Esq, height=0.05, distance=10)
    Q_factors = []
    res_wvls = []

    for peak in peaks:
        left_bound = max(0, peak - 20)
        right_bound = min(len(freqs) - 1, peak + 20)
        fit_freqs = freqs[left_bound:right_bound]
        fit_Esq = Esq[left_bound:right_bound]

        # Initial guess for fitting parameters
        x0_guess = freqs[peak]
        A_guess = max(fit_Esq)
        gamma_guess = (fit_freqs[-1] - fit_freqs[0]) / 4  # Initial guess for FWHM

        # Perform curve fitting
        popt, _ = curve_fit(lorentzian, fit_freqs, fit_Esq, p0=[x0_guess, gamma_guess, A_guess])
        # Extract parameters   
        x0_fit, gamma_fit, A_fit = popt

        # Calculate FWHM from gamma_fit (for Lorentzian, FWHM = 2 * gamma)
        fwhm = 2 * gamma_fit

        # Calculate Q-factor
        q_factor = x0_fit / fwhm
        Q_factors.append(q_factor)
        res_wvls.append(c / x0_fit)

    return Q_factors, res_wvls, peaks

def compute_fft(E, dt, pad_factor=1):
    N = len(E)
    N_padded = N*pad_factor
    E_padded = np.pad(E, (0, N_padded - N), mode='constant')  # Pad with zeros
    E_fft = np.fft.fft(E_padded)
    E_fft = np.fft.fftshift(E_fft)  # Shift zero frequency components to the center
    freqs = np.fft.fftfreq(N_padded, dt)
    freqs = np.fft.fftshift(freqs)  # Shift zero frequency components to the center

    return freqs, E_fft

#Microdisk resonances
def microdisk_resonances(disk_radius, disk_thickness, material_index, wavelength, wavelength_span,
                         sub_disk = 0, sub_disk_radius = 0):
    
    sim = lp.FDTD(hide=True)

    output_folder = "r_" + str(int(disk_radius*1e9)) + "nm_t_" + str(int(disk_thickness*1e9)) + "nm"
    FDTD_zmin_BC = "symmetric"
    if sub_disk == 1:
        subdisk = so.microdisk(thickness = disk_thickness/2 + wavelength,
                                z = -disk_thickness/2 - (disk_thickness/2 + wavelength)/2,
                                radius = sub_disk_radius, index = material_index,
                                name = "sub_disk")
        subdisk.add_to_sim(sim)
        FDTD_zmin_BC = "PML"
        output_folder = ("r_" + str(int(disk_radius*1e9)) + "nm_t_" + 
                         str(int(disk_thickness*1e9))  + "nm_U_" + 
                         str(int((disk_radius - sub_disk_radius)*1e9)) + "nm")
    
    os.makedirs(output_folder, exist_ok=True)

    sim.setglobalsource("wavelength start", wavelength - wavelength_span/2)
    sim.setglobalsource("wavelength stop", wavelength + wavelength_span/2)
    sim.setglobalmonitor("wavelength center", wavelength)
    sim.setglobalmonitor("wavelength span", 2*wavelength_span)

    disk = so.microdisk(thickness = disk_thickness, radius = disk_radius, index = material_index)
    disk.add_to_sim(sim)

    if disk_thickness == 0:
        dimension = '2D'
        dipole_angle = 90
    else:
        dimension = '3D'
        dipole_angle = 0
    fdtd = so.FDTD(xspan = disk_radius*2 + 2*wavelength, 
                   yspan = disk_radius*2 + 2*wavelength, 
                   zspan = disk_thickness + 2*wavelength, 
                   dimension = dimension, 
                   sim_time = 6e-12,
                   xmin_bc = 'symmetric',
                   ymin_bc = 'anti-symmetric',
                   zmin_bc = FDTD_zmin_BC,
                   mesh_accuracy = 3)
    fdtd.add_to_sim(sim)

    theta = 25
    for xi in [-1,0,1]:
        for yi in [-1,0,1]:
            x = np.cos(theta*np.pi/180)*disk_radius - 3*wavelength/32 + xi*wavelength/16
            y = np.sin(theta*np.pi/180)*disk_radius - 3*wavelength/32 + yi*wavelength/16
            dipole = so.dipole(x=x, y=y, z = 5e-9, wvl_start = wavelength - wavelength_span/2, wvl_stop = wavelength + wavelength_span/2, 
                               dipole_type = 'Magnetic Dipole', theta=dipole_angle, phi=dipole_angle)
            dipole.add_to_sim(sim)

    theta = 35
    x = np.cos(theta*np.pi/180)*(disk_radius - wavelength/8)
    y = np.sin(theta*np.pi/180)*(disk_radius - wavelength/8)
    time_monitor = so.time_monitor(x=x, y = y, z = 5e-9, start_time = 500e-15, min_sampling = 100)
    time_monitor.add_to_sim(sim)
    
    mesh = so.mesh(xspan = 2*disk_radius, yspan = 2*disk_radius)
    #mesh.add_to_sim(sim)

    if os.path.isfile('gui.fps'):
        os.remove('gui.fsp')
    sim.save('gui.fsp')
    
    try:
        sim.run()
        
        time_spectrum = sim.getresult(time_monitor.get_name(), 'E')#sim.getresult(time_monitor.get_name(), 'spectrum')
        E_time = time_spectrum['E'][0,0,0,:,:]
        t      = time_spectrum['t'][:,0]
        Ex = E_time[:,0]
        Ey = E_time[:,1]
        Ez = E_time[:,2]
        dt = t[1] - t[0]

        freqs_x, E_fft_x = compute_fft(Ex, dt, 100)
        freqs_y, E_fft_y = compute_fft(Ey, dt, 100)
        freqs_z, E_fft_z = compute_fft(Ez, dt, 100)

        E_fft_combined = np.sqrt(np.abs(E_fft_x)**2 + np.abs(E_fft_y)**2 + np.abs(E_fft_z)**2)
        E_power_spectrum = E_fft_combined**2
        E_power_spectrum = E_power_spectrum / np.max(E_power_spectrum)
        freqs = freqs_x
        with np.errstate(divide='ignore', invalid='ignore'):
            wvls = np.where(freqs != 0, c / freqs, np.inf)  # Avoid divide by zero

        peaks,_ = find_peaks(E_power_spectrum, height=0.02, distance=1) #
        all_res_wvls = wvls[peaks]
        res_wvls = np.array(all_res_wvls[np.abs(all_res_wvls - wavelength) < wavelength_span])
        closest_res_wvl = all_res_wvls[np.argmin(np.abs(all_res_wvls - wavelength))]

        plt.figure()
        plt.plot(wvls, E_power_spectrum)
        plt.scatter(all_res_wvls, E_power_spectrum[peaks], c='red')
        plt.title('Power Spectrum |E|^2')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Power')
        plt.xlim((1500e-9, 1600e-9))
        plt.savefig(os.path.join(output_folder, 'FFT.png'))
        plt.close()
        

        if len(res_wvls) == 0:
            res_wvls = np.array([closest_res_wvl])

        if len(res_wvls) > 0:
            sim.switchtolayout()

            theta = 45
            x = np.cos(theta*np.pi/180)*(disk_radius - wavelength/16 - 5e-9)
            y = np.sin(theta*np.pi/180)*(disk_radius - wavelength/16 - 5e-9)
            Qanalysis = so.Qanalysis(x=x, y=y, z=wavelength/32+5e-9, 
                                    xspan = wavelength/8, yspan = wavelength/8, zspan = wavelength/16,
                                    nx=2, ny=2, nz = 2,
                                    fmin = c/(np.max(res_wvls) + wavelength_span/2), fmax = c/(np.min(res_wvls)-wavelength_span/2),
                                    start_time = 500e-15)

            Qanalysis.add_to_sim(sim)

            sim.run()
            try:
                sim.runanalysis()
                Qvals = sim.getresult(Qanalysis.get_name(), 'Q')
                Q = Qvals['Q']
                Q_res_wvls = Qvals['lambda']

                plt.figure(figsize=(8, 6))
                plt.plot(Q_res_wvls*1e9, Q, 'o')
                plt.xlabel('resonance wavelength (nm)')
                plt.ylabel('Q')
                plt.grid(True)
                plt.yscale('log')
                plt.savefig(os.path.join(output_folder, 'Q_vs_wvl.png'))
                plt.close()     
        
                Q_res_wvls = np.append(res_wvls, Q_res_wvls)
                Q_res_wvls = np.unique(Q_res_wvls)
            except Exception as e:
                print(f"Error: {e}")

            Q_res_wvls = res_wvls
            sim.switchtolayout()
            DFT_monitor_list = [None] * len(Q_res_wvls)
            for i in range(len(Q_res_wvls)):
                res = Q_res_wvls[i]
                DFT_monitor_list[i] = so.DFT_monitor(type='2D Z-normal',
                                            source_limits = 0,
                                            x=0, 
                                            y=0,
                                            z = 5e-9, 
                                            xspan = disk_radius*2 + 1.5e-6, 
                                            yspan = disk_radius*2 + 1.5e-6,
                                            num_freqs = 1, 
                                            wvl_center = res, 
                                            apodization = 'full',
                                            apodization_center = 3500e-15,
                                            apodization_width = 2000e-15)
                DFT_monitor_list[i].add_to_sim(sim)
            
            theta = 55
            x = np.cos(theta*np.pi/180)*(disk_radius - wavelength/8)
            y = np.sin(theta*np.pi/180)*(disk_radius - wavelength/8)
            DFT_monitor = so.DFT_monitor(x=x, y=y, z = 5e-9, num_freqs = 5000,
                                         wvl_center = np.mean(Q_res_wvls), wvl_span = 3*wavelength_span,
                                         source_limits = 0,
                                         apodization = 'full', apodization_center = 3500e-15, apodization_width = 2000e-15)
            DFT_monitor.add_to_sim(sim)

            sim.run()

            decay_lengths = np.zeros_like(Q_res_wvls)
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

                x_pos = x[x0+20:-20]
                y_pos = y[y0+20:-20]

                sliced_E_sq = E_sq[x0+20:-20, y0+20:-20]
                
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
                plt.savefig(os.path.join(output_folder, 'Esq_' + str(int(Q_res_wvls[i]*1e9)) + 'nm_.png'))
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
                plt.savefig(os.path.join(output_folder, 'E_1D_' + str(int(Q_res_wvls[i]*1e9)) + 'nm_.png'))
                plt.close()
                decay_lengths[i] = disk_radius - r_1d[index]

                i = i + 1

            spectrum = sim.getresult(DFT_monitor.get_name(), 'E')
            E = spectrum['E'][0,0,0,:,:]
            wvl = spectrum['lambda'][:,0]
            freqs = spectrum['f'][:,0]
            
            Esq = (np.sqrt(np.abs(np.sum(E * np.conjugate(E), axis = 1))))**2
            Esq = Esq / np.max(Esq)
            
            plt.figure(figsize=(8, 6))
            plt.plot(wvl*1e9, Esq, label='|E| vs wvl')
            plt.vlines(x=Q_res_wvls*1e9, ymin=np.zeros_like(Q_res_wvls), ymax=np.ones_like(Q_res_wvls), colors='r', linestyles='dashed')
            plt.xlabel('resonance wavelength (nm)')
            plt.ylabel('|E|^2')
            plt.grid(True)
            plt.savefig(os.path.join(output_folder, 'E_vs_wvl.png'))
            plt.close()

        results = {'resonance_wavelengths': res_wvls,
                    'Q_factors' : Q,
                    'Q_res_wvls': Q_res_wvls,
                    'decay_lengths': decay_lengths,
                    'Efield': Esq,
                    'wvl': wvl,
                    'Esq_time': E_power_spectrum,
                    'wvl_time': wvls}
        

        with open(os.path.join(output_folder, 'output.p'), 'wb') as f:
            pickle.dump(results, f)
        scipy.io.savemat(os.path.join(output_folder, 'output.mat'), results)
        return results

    except Exception as e:
        print(e)    
        return -1
        
def microdisk_coupler(disk_radius, disk_thickness, material_index, coupler_width, coupler_gap, wavelength):
    sim = lp.FDTD(hide=True)

    output_folder = "coupler_width_" + str(int(coupler_width*1e9)) + "nm_gap_" + str(int(coupler_gap*1e9)) + "nm"
    os.makedirs(output_folder, exist_ok=True)

    sim.setglobalsource("wavelength start", wavelength)
    sim.setglobalsource("wavelength stop", wavelength)
    sim.setglobalmonitor("wavelength center", wavelength)
    sim.setglobalmonitor("wavelength span", 0)

    fdtd = so.FDTD(xspan = disk_radius + 2*wavelength, 
                   yspan = coupler_width*2 + 2*wavelength, 
                   zspan = disk_thickness + 2*wavelength, 
                   dimension = "3D", 
                   sim_time = 6e-12,
                   xmin_bc = 'PML',
                   ymin_bc = 'PML',
                   zmin_bc = 'PML',
                   mesh_accuracy = 1)
    fdtd.add_to_sim(sim)

    coupler = so.waveguide(x = 0,
                           y = 0,
                           z = 0,
                           wx = disk_radius,
                           wy = coupler_width,
                           wz = disk_thickness,
                           index = material_index)
    coupler.add_to_sim(sim)

    disk = so.microdisk(thickness = disk_thickness, radius = disk_radius, index = material_index,
                        x = 0, z = 0, y = disk_radius + coupler_gap + coupler_width/2)
    disk.add_to_sim(sim)

    input_port = so.port(name = 'Input', x = -disk_radius/2 + wavelength/2, y = 0, z = 0, 
                          yspan = 1.5*wavelength, zspan = 1.5*wavelength)
    input_port.add_to_sim(sim)

    if os.path.isfile('gui.fps'):
        os.remove('gui.fsp')
    sim.save('gui.fsp')
    
    #try:
    #    sim.run()

        

