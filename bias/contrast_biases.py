import matplotlib.pyplot as plt
import numpy as np
import os

from refnx.analysis import Objective, GlobalObjective, CurveFitter

from structures import Bilayer
from simulate import simulate

def contrast_biases(initial_contrast, new_contrasts, angle_times, n=50):
    DMPC = Bilayer()
    xi = DMPC.parameters
    
    for param in xi:
        param.bounds = (param.value*0.5, param.value*1.5)
    
    objective_initial = Objective(*simulate(DMPC.using_contrast(initial_contrast), angle_times))
    
    true = np.asarray([param.value for param in xi])
    biases = []
    for x, new_contrast in enumerate(new_contrasts, 1):
        fitted = []
        for i in range(n):
            for param in xi:
                param.value = np.random.uniform(param.bounds.lb, param.bounds.ub)
            
            objective_new = Objective(*simulate(DMPC.using_contrast(new_contrast), angle_times))
            
            fitter = CurveFitter(GlobalObjective([objective_initial, objective_new]))
            fitter.fit('differential_evolution', verbose=False)
            fitted.append([param.value for param in xi])
    
        biases.append(np.mean(fitted, axis=0) - true)
        print("{0}/{1}".format(x, len(new_contrasts)))
    
    fig = plt.figure(figsize=[9,7], dpi=600)
    ax = fig.add_subplot(111)
    
    biases = np.asarray(biases)
    for i, param in enumerate(xi):
        ax.plot(contrasts, biases[:,i], label=param.name)

    ax.set_xlabel("$\mathregular{Contrast\ SLD\ (10^{-6} \AA^{-2})}$", fontsize=11, weight='bold')
    ax.set_ylabel("Bias", fontsize=11, weight='bold')
    ax.legend()
    
    save_path = "./results/contrast_biases"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    fig.savefig(save_path+"/DMPC.png")

if __name__ == "__main__":
   angle_times = {0.7: (70, 50), #Angle: (Points, Time)
                  2.0: (70, 200)}
   initial = 6.36
   contrasts = np.arange(-0.56, 6.36, 0.05)
   contrast_biases(initial, contrasts, angle_times, n=50)