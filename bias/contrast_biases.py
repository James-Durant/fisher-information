import matplotlib.pyplot as plt
import numpy as np
import os

from refnx.analysis import Objective, GlobalObjective, CurveFitter

from structures import Bilayer
from simulate import simulate

def bias_derivative(initial_contrast, new_contrasts, angle_times, n=50):
    x = 4
    tailgroup_range = np.arange(12, 22, 1)
    tailgroup_biases = []
    for value in tailgroup_range:
        DMPC = Bilayer()
        xi = DMPC.parameters
        xi[x].value = value
        
        for i, param in enumerate(xi):
            if i == x:
                param.setp(vary=True, bounds=(value*0.75, value*1.25))
            else:
                param.setp(vary=False)
    
        objective_initial = Objective(*simulate(DMPC.using_contrast(initial_contrast), angle_times))
        
        true = xi[x].value
        biases = []
        for k, new_contrast in enumerate(new_contrasts, 1):
            fitted = []
            for _ in range(n):                
                objective_new = Objective(*simulate(DMPC.using_contrast(new_contrast), angle_times))
                xi[x].value = np.random.uniform(xi[x].bounds.lb, xi[x].bounds.ub)
                
                fitter = CurveFitter(GlobalObjective([objective_initial, objective_new]))
                fitter.fit('differential_evolution', verbose=False)
                fitted.append(xi[x].value)
        
            biases.append(np.mean(fitted) - true)
            print("{0}/{1}".format(k, len(new_contrasts)))
        
        tailgroup_biases.append(biases)
        print()
        
    fig = plt.figure(figsize=[9,7], dpi=600)
    ax = fig.add_subplot(111)
    
    for i, value in enumerate(tailgroup_range):
        ax.plot(contrasts, tailgroup_biases[i], label="TG_thick="+str(value))
    
    ax.set_xlabel("$\mathregular{Contrast\ SLD\ (10^{-6} \AA^{-2})}$", fontsize=11, weight='bold')
    ax.set_ylabel("$\mathregular{Bias\ (\AA)}$", fontsize=11, weight='bold')
    ax.legend(loc='upper right')
    
    save_path = "./results/contrast_biases"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    fig.savefig(save_path+"/tailgroup_bias_derivative.png")     

def contrast_biases(initial_contrast, new_contrasts, angle_times, n=10):
    DMPC = Bilayer()
    xi = DMPC.parameters
    
    for param in xi:
        param.bounds = (param.value*0.5, param.value*1.5)
    
    objective_initial = Objective(*simulate(DMPC.using_contrast(initial_contrast), angle_times))
    
    true = np.asarray([param.value for param in xi])
    biases = []
    for x, new_contrast in enumerate(new_contrasts, 1):
        fitted = []
        for _ in range(n):  
            for i, param in enumerate(xi):
                param.value = true[i]
          
            objective_new = Objective(*simulate(DMPC.using_contrast(new_contrast), angle_times))
            
            for param in xi:
                param.value = np.random.uniform(param.bounds.lb, param.bounds.ub)

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
   angle_times = {0.7: (70, 5), #Angle: (Points, Time)
                  2.0: (70, 20)}
   initial = 6.36
   
   #contrasts = np.arange(-0.56, 6.36, 0.05)
   #contrast_biases(initial, contrasts, angle_times, n=50)
   
   contrasts = np.arange(-0.56, 6.36, 0.25)
   bias_derivative(initial, contrasts, angle_times, n=50)