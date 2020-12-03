import numpy as np
import matplotlib.pyplot as plt

from refnx.reflect import ReflectModel

from generate   import SCALE, DQ, BKG, BKG_RATE
from structures import easy_sample_1, many_param_sample, multiple_contrast_sample
from structures import thin_layer_sample_1, thin_layer_sample_2
from structures import similar_sld_sample_1, similar_sld_sample_2

def simulate(structure, angle, ax=None, points=200, time=1, file="../utils/directbeam_wavelength.dat"):
    model = ReflectModel(structure, scale=SCALE, dq=DQ, bkg=BKG)
    
    direct_beam  = np.loadtxt(file, delimiter=',')
    wavelengths  = direct_beam[:,0]
    flux_density = direct_beam[:,1]*time*((angle/0.3)**2) #Fix this
    
    theta = angle*np.pi/180 #Angle in radians
    q = 4*np.pi*np.sin(theta) / wavelengths
    log_q = np.logspace(np.log10(np.min(q)), np.log10(np.max(q)), points+1)
    flux_binned, q_bin_edges = np.histogram(q, log_q, weights=flux_density)
    
    #Get the bin centres
    q_binned = [(q_bin_edges[i] + q_bin_edges[i+1]) / 2 for i in range(points)]

    r, r_errors = [], []
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
        plt.xlabel("$\mathregular{Q\ (Å^{-1})}$", fontsize=11, weight='bold')
        plt.ylabel('Reflectivity (arb.)',         fontsize=11, weight='bold')
        plt.yscale('log')
        plt.xlim(0, 0.25)
    
    return q_binned, r, r_errors, flux_binned, model
    
    
if __name__ == "__main__":
    structures = many_param_sample()
    fig = plt.figure(figsize=[9,7])
    ax = fig.add_subplot(111)
    for structure in structures:
        for angle in [0.7, 2]:
            simulate(structure, angle, ax=ax, points=100, time=0.8)
    plt.show()
        
    