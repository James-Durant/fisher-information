import numpy as np
import matplotlib.pyplot as plt

from refnx.reflect import ReflectModel

DQ       = 2
SCALE    = 1
BKG      = 1e-6
BKG_RATE = 5e-7

def simulate_noisy(structure, angle, points, time, save_path=None, file="../simulation/directbeam_wavelength.dat", ax=None):    
    direct_beam  = np.loadtxt(file, delimiter=',')
    wavelengths  = direct_beam[:,0]
    flux_density = direct_beam[:,1]*time#*((angle/0.3)**2)
    
    theta = angle*np.pi/180 #Angle in radians
    q = 4*np.pi*np.sin(theta) / wavelengths
    log_q = np.logspace(np.log10(np.min(q)), np.log10(np.max(q)), points+1)
    flux_binned, q_bin_edges = np.histogram(q, log_q, weights=flux_density)
    
    #Get the bin centres
    q_binned = [(q_bin_edges[i] + q_bin_edges[i+1]) / 2 for i in range(points)]
    r, r_errors = [], []
    
    model = ReflectModel(structure, scale=SCALE, dq=DQ, bkg=BKG)
    
    for i in range(points):
        q_point = q_binned[i]
        flux    = flux_binned[i]
        
        r_point = model(q_point) + max(np.random.normal(1, 0.5)*BKG_RATE, 0)
        
        count       = r_point*flux
        error_bar   = np.sqrt(count)
        noisy_count = np.random.normal(loc=count, scale=error_bar)
        
        r_noisy = noisy_count / flux
        r_error = error_bar / flux
        
        r.append(r_noisy)
        r_errors.append(r_error)
    
    if ax:
        ax.errorbar(q_binned, r, r_errors, marker="o", ms=3, lw=0, elinewidth=1, capsize=1.5)
        ax.set_xlabel("$\mathregular{Q\ (Å^{-1})}$", fontsize=11, weight='bold')
        ax.set_ylabel('Reflectivity (arb.)',         fontsize=11, weight='bold')
        ax.set_yscale('log')
        ax.set_xlim(0, 0.25)
        
    if save_path:
        data = np.zeros((points, 3))
        data[:,0] = q_binned
        data[:,1] = r
        data[:,2] = r_errors
        np.savetxt(save_path+"/data.dat", data, delimiter=",")
    
    return model, q_binned, r, r_errors, flux_binned

def vary_model(model):
    for component in model.structure.components[1:-1]: #Skip over Air/D20 and substrate
        sld_bounds = (component.sld.real.value*0.5, component.sld.real.value*1.5)
        component.sld.real.setp(vary=True, bounds=sld_bounds)

        thick_bounds = (component.thick.value*0.5, component.thick.value*1.5)
        component.thick.setp(vary=True, bounds=thick_bounds)

        #Set the SLD and thickness to arbitrary initial values (within their bounds).
        component.sld.real.value = np.random.uniform(*sld_bounds)
        component.thick.value    = np.random.uniform(*thick_bounds)

def plot_objective(objective):
    fig = plt.figure(figsize=[9,7])
    ax = fig.add_subplot(111)

    y, y_err, model = objective._data_transform(model=objective.generative())
    # Add the data in a transformed fashion.
    ax.errorbar(objective.data.x, y, y_err, marker="o", ms=3, lw=0, elinewidth=1, capsize=1.5)
    #Add the prediction/fit
    ax.plot(objective.data.x, model, color="red", zorder=20)

    plt.xlabel("$\mathregular{Q\ (Å^{-1})}$", fontsize=11, weight='bold')
    plt.ylabel('Reflectivity (arb.)',         fontsize=11, weight='bold')
    plt.yscale('log')
    plt.show()
    
if __name__ == "__main__":
    from structures import thin_layer_sample_1,  thin_layer_sample_2
    from structures import similar_sld_sample_1, similar_sld_sample_2
    from structures import easy_sample, many_param_sample, multiple_contrast_sample
    
    fig = plt.figure(figsize=[9,7])
    ax  = fig.add_subplot(111)
    
    structures = easy_sample()
    points     = 250
    time       = 100
    angles     = [0.7, 2]
    
    for structure in structures:
        for angle in angles:
            simulate_noisy(structure, angle, points, time, ax=ax)
    plt.show()