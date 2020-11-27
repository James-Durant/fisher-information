import sys
sys.path.append("../utils") # Adds higher directory to python modules path.

import numpy as np
import matplotlib.pyplot as plt

from refnx.dataset  import ReflectDataset
from refnx.reflect  import ReflectModel
from refnx.analysis import Objective

from generate   import plot_objective, SCALE, DQ, BKG, BKG_RATE
from structures import easy_sample_1, many_param_sample
from structures import thin_layer_sample_1, thin_layer_sample_2
from structures import similar_sld_sample_1, similar_sld_sample_2

def simulate(structure, angle, ax, time=1, file="../utils/directbeam_wavelength.dat"):
    theta = angle*np.pi/180 #Angle in radians
    model = ReflectModel(structure, scale=SCALE, dq=DQ, bkg=BKG)
    q, r, r_errors = [], [], []
    
    direct_beam  = np.loadtxt(file, delimiter=',')
    wavelengths  = direct_beam[:,0]
    flux_density = direct_beam[:,1]*time
    
    for i in range(len(direct_beam)):
        wavelength = wavelengths[i]
        flux       = flux_density[i]
        
        q_point = 4*np.pi*np.sin(theta) / wavelength
        r_point = model(q_point) + max(np.random.normal(1, 0.5)*BKG_RATE, 0)
        
        count       = r_point*flux
        error_bar   = np.sqrt(count)
        noisy_count = np.random.normal(loc=count, scale=error_bar)
        
        r_noisy = noisy_count / flux
        r_error = error_bar / flux
        
        q.append(q_point)
        r.append(r_noisy)
        r_errors.append(r_error)
    
    data = ReflectDataset([q,r,r_errors])
    objective = Objective(model, data)

    y, y_err, model = objective._data_transform(model=objective.generative())
    # Add the data in a transformed fashion.
    ax.errorbar(objective.data.x, y, y_err, marker="o", ms=3, lw=0, elinewidth=1, capsize=1.5)
    #Add the prediction/fit
    #ax.plot(objective.data.x, model, color="red", zorder=20)

    plt.xlabel("$\mathregular{Q\ (Å^{-1})}$", fontsize=11, weight='bold')
    plt.ylabel('Reflectivity (arb.)',         fontsize=11, weight='bold')
    plt.yscale('log')
    
    return q, r, r_errors, flux_density
    
    
if __name__ == "__main__":
    structure = thin_layer_sample_1()
    fig = plt.figure(figsize=[9,7])
    ax = fig.add_subplot(111)
    for angle in [0.7, 1.5]:
        simulate(*structure, angle, ax, time=1)
    plt.show()
        
    