import sys, copy
sys.path.append("../utils") #Adds higher directory to python modules path.

import numpy as np
import matplotlib.pyplot as plt

from refnx.dataset  import ReflectDataset
from refnx.analysis import Objective, CurveFitter

from generate    import generate_noisy_single
from structures  import thin_layer_sample_1, thin_layer_sample_2
from structures  import similar_sld_sample_1, similar_sld_sample_2
from structures  import easy_sample_1, many_param_sample

def gradient(model, parameter, q_point, step=0.005):
    step = parameter.value * step #0.5% step by default
    old = parameter.value

    x1 = parameter.value = old - step #First point
    y1 = model(q_point)

    x2 = parameter.value = old + step #Second point
    y2 = model(q_point) #Get new r value with changed model

    parameter.value = old #Reset parameter
    return (y2-y1) / (x2-x1) #Return the gradient

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

def compare_fit_variance(structure, n):
    param_estimates = []
    for i in range(n):
        if i > 0 and i % 100 == 0:
            print("{0}/{1}...".format(i, n))
        structure_copy = copy.deepcopy(structure)   
        model, data, r, flux = generate_noisy_single(structure_copy)
        objective = Objective(model, ReflectDataset(data))
        fitter = CurveFitter(objective)
        fitter.fit('L-BFGS-B', verbose=False)
        xi = objective.varying_parameters()
        param_estimates.append([param.value for param in xi])
    
    g = calc_FIM(data[0], r, xi, flux, model)
    print("Fisher Information: ", np.diag(g))
    print("Inverse Fisher Information: ", 1 / np.diag(g))
    
    variances = np.var(np.array(param_estimates), axis=0)
    print("Parameter Estimation Variances: ", variances)

def compare_errors(structure):
    noise_constants = 10**np.arange(1, 5, 0.5, dtype=float)
    fit_errors = []
    fisher_errors = []
    for noise_constant in noise_constants:
        model, data, r, flux = generate_noisy_single(structure, noise_constant)
        q = data[0]

        objective = Objective(model, ReflectDataset(data))
        fitter = CurveFitter(objective)
        fitter.fit('differential_evolution', polish=False, verbose=False)
        fitter.fit('L-BFGS-B', verbose=False)
    
        xi = objective.varying_parameters()
        fit_error = [param.stderr for param in xi]
        g = calc_FIM(q, r, xi, flux, model)
        fisher_error = 1 / np.sqrt(np.diag(g))
        
        fit_errors.append(fit_error)
        fisher_errors.append(fisher_error)
        
    fit_errors = np.array(fit_errors)
    fisher_errors = np.array(fisher_errors)
    
    for i in range(fit_errors.shape[1]):
        plt.plot(np.log(noise_constants), np.log(fit_errors[:,i]))
        plt.show()
        
        plt.plot(np.log(noise_constants), np.log(fisher_errors[:,i]))
        plt.show()
    
    
if __name__ == "__main__":
    """
    Functions for getting structures:
        easy_sample_1
        thin_layer_sample_1
        thin_layer_sample_2
        similar_sld_sample_1
        similar_sld_sample_2
        many_param_sample
    """
    structure = easy_sample_1()
    compare_errors(*structure)
    #compare_fit_variance(*structure, 1000)
