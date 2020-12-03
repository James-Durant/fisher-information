import sys, copy
sys.path.append("../simulation") #Adds higher directory to python modules path.

import numpy as np
import matplotlib.pyplot as plt

from refnx.dataset  import ReflectDataset
from refnx.analysis import Objective, CurveFitter

from simulate import simulate_noisy, vary_model, plot_objective

def calc_FIM(q, r, xi, flux, model):
    n = len(r)
    m = len(xi)
    J = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            J[i,j] = gradient(model, xi[j], q[i])

    M = np.diag(flux/r, k=0)
    g = np.dot(np.dot(J.T, M), J)
    return g

def gradient(model, parameter, q_point, step=0.005):
    step = parameter.value * step #0.5% step by default
    old = parameter.value

    x1 = parameter.value = old - step #First point
    y1 = model(q_point)

    x2 = parameter.value = old + step #Second point
    y2 = model(q_point) #Get new r value with changed model

    parameter.value = old #Reset parameter
    return (y2-y1) / (x2-x1) #Return the gradient

def compare_fit_variance(structure, angle, points, time, n=1000):
    param_estimates = []
    for i in range(n):
        if i > 0 and i % 100 == 0:
            print("{0}/{1}...".format(i, n))
            
        structure_copy = copy.deepcopy(structure)   
        model, q, r, r_error, flux = simulate_noisy(structure_copy, angle, points, time)
        vary_model(model)
        
        objective = Objective(model, ReflectDataset([q, r, r_error]))
        fitter = CurveFitter(objective)
        fitter.fit('L-BFGS-B', verbose=False)
        
        xi = objective.varying_parameters()
        param_estimates.append([param.value for param in xi])
    
    g = calc_FIM(q, r, xi, flux, model)
    print("Fisher Information: ", np.diag(g))
    print("Inverse Fisher Information: ", 1 / np.diag(g))
    
    variances = np.var(np.array(param_estimates), axis=0)
    print("Parameter Estimation Variances: ", variances)

def compare_errors(structure, angle, points, time_constants):
    fit_errors = []
    fisher_errors = []
    
    for i, time_constant in enumerate(time_constants, 1):
        print(">>> Fitting {0}/{1}...".format(i, len(time_constants)))
        
        model, q, r, r_error, flux = simulate_noisy(structure, angle, points, time_constant)
        vary_model(model)

        objective = Objective(model, ReflectDataset([q, r, r_error]))
        fitter = CurveFitter(objective)
        fitter.fit('differential_evolution', polish=False, verbose=False)
        fitter.fit('L-BFGS-B', verbose=False)
        
        #plot_objective(objective)
    
        xi = objective.varying_parameters()
        fit_error = [param.stderr for param in xi]
        g = calc_FIM(q, r, xi, flux, model)
        fisher_error = 1 / np.sqrt(np.diag(g))
        
        fit_errors.append(fit_error)
        fisher_errors.append(fisher_error)
        
    fit_errors = np.array(fit_errors)
    fit_fig    = plt.figure(num=1, dpi=600)
    fit_ax     = fit_fig.add_subplot(111)
    fit_ax.set_xlabel("Log Time Constant")
    fit_ax.set_ylabel("Log L-BFGS-B Error")
    
    fisher_errors = np.array(fisher_errors)
    fisher_fig    = plt.figure(num=2, dpi=600)
    fisher_ax     = fisher_fig.add_subplot(111)
    fisher_ax.set_xlabel("Log Time Constant")
    fisher_ax.set_ylabel("Log Fisher Error")

    for i in range(len(xi)):
        param = xi[i].name
        fit_ax.plot(np.log(time_constants), np.log(fit_errors[:,i]), label=param)
        fisher_ax.plot(np.log(time_constants), np.log(fisher_errors[:,i]), label=param)
    
    fit_ax.legend()
    fisher_ax.legend()
    plt.show()

if __name__ == "__main__":
    from structures import similar_sld_sample_1, similar_sld_sample_2
    from structures import thin_layer_sample_1,  thin_layer_sample_2
    from structures import easy_sample, many_param_sample
    
    structure = easy_sample()
    points    = 200
    angle     = 0.7
    
    time_constants = 10**np.arange(0.5, 5, 0.5, dtype=float)
    compare_errors(*structure, angle, points, time_constants)
    
    time = 100
    #compare_fit_variance(*structure, angle, points, time, n=100)
