import sys, copy
sys.path.append("../simulation") #Adds higher directory to python modules path.

import numpy as np
import matplotlib.pyplot as plt

from refnx.dataset  import ReflectDataset
from refnx.analysis import Objective, CurveFitter

from simulate import simulate_noisy, vary_model, plot_objective

def calc_FIM(q, r, xi, flux, model):
    """Calculates the Fisher information matrix for given a dataset and model.

    Args:
        q (numpy.ndarray): array of Q values.
        r (numpy.ndarray): array of reflectivity values corresponding to each Q value.
        xi (list): list of refnx Parameter objects representing each varying parameter value.
        flux (numpy.ndarray): array of flux values corresponding to each Q value.
        model (refnx.reflect.ReflectModel): the model to calculate the gradient with.

    Returns:
        numpy.ndarray: Fisher information matrix.

    """
    n = len(r)
    m = len(xi)
    J = np.zeros((n,m))
    #Calculate the gradient of the model reflectivity (r) with every
    #model parameter (ξ) for every model data point.
    for i in range(n):
        for j in range(m):
            J[i,j] = gradient(model, xi[j], q[i])

    M = np.diag(flux/r, k=0)
    g = np.dot(np.dot(J.T, M), J)
    return g

def gradient(model, parameter, q_point, step=0.005):
    """Calculate a two-point gradient of model reflectivity with model `parameter`.

    Args:
        model (refnx.reflect.ReflectModel): the model to calculate the gradient with.
        parameter (refnx.analysis.Parameter): the parameter to vary when calculating the gradient.
        q_point (float): the Q value of the model reflectivity point to calculate the gradient of.
        step (float): the step size to take when calculating the gradient.

    Returns:
        float: a two-point gradient.

    """
    step = parameter.value * step #0.5% step by default
    old = parameter.value

    x1 = parameter.value = old - step #First point
    y1 = model(q_point) #Get new r value with altered model.

    x2 = parameter.value = old + step #Second point
    y2 = model(q_point)

    parameter.value = old #Reset parameter
    return (y2-y1) / (x2-x1) #Return the gradient

def compare_fit_variance(structure, angle, points, time, n=1000):
    """Compares the multiplicative inverse Fisher information (variance) with
       variance in parameter estimation using traditional fitting for `n` fits.

    Args:
        structure (refnx.reflect.Structure): the structure to simulate the experiment for.
        angle (float): measurement angle to use in the experiment simulation.
        points (int): number of points to obtain from binning in the experiment simulation.
        time (int): how long to measure for in the experiment simulation.
        n (int): number of fits to run.

    """
    param_estimates = []
    for i in range(n): #Fit `n` times.
        if i > 0 and i % 100 == 0: #Display progress every 100 fits.
            print("{0}/{1}...".format(i, n))

        #Copy the structure to avoid using previous fits' results.
        structure_copy = copy.deepcopy(structure)
        #Simulate the experiment using the given angle, number of points and time.
        model, q, r, r_error, flux = simulate_noisy(structure_copy, angle, points, time)
        vary_model(model) #Vary the model and initialise parameters to random values.

        #Fit the model using L-BFGS-B.
        objective = Objective(model, ReflectDataset([q, r, r_error]))
        fitter = CurveFitter(objective)
        fitter.fit('L-BFGS-B', verbose=False)

        xi = objective.varying_parameters() #Get the parameter estimates.
        param_estimates.append([param.value for param in xi])

    g = calc_FIM(q, r, xi, flux, model) #Calculate the Fisher information matrix.
    print("Fisher Information: ", np.diag(g))
    print("Inverse Fisher Information: ", 1 / np.diag(g))

    #Calculate the variances in parameter estimates from `n` fits.
    variances = np.var(np.array(param_estimates), axis=0)
    print("Parameter Estimation Variances: ", variances)

def compare_errors(structure, angle, points, time_constants):
    """Compares traditional fitting errors and Fisher errors with increasing time.

    Args:
        structure (refnx.reflect.Structure): the structure to simulate the experiment for.
        angle (float): measurement angle to use in the experiment simulation.
        points (int): number of points to obtain from binning in the experiment simulation.
        time_constants (numpy.ndarray): array of time constants to use for the comparison.

    """
    fit_errors, fisher_errors = [], []
    for i, time_constant in enumerate(time_constants, 1):
        print(">>> Fitting {0}/{1}...".format(i, len(time_constants))) #Display progress

        #Simulate the experiment using the given angle, number of points and time.
        model, q, r, r_error, flux = simulate_noisy(structure, angle, points, time_constant)
        vary_model(model) #Vary the model and initialise parameters to random values.

        #Fit the data using differential evolution / L-BFGS-B
        objective = Objective(model, ReflectDataset([q, r, r_error]))
        fitter = CurveFitter(objective)
        fitter.fit('differential_evolution', polish=False, verbose=False)
        fitter.fit('L-BFGS-B', verbose=False)
        #plot_objective(objective)

        xi = objective.varying_parameters()
        fit_error = [param.stderr for param in xi] #Get the fitting errors.
        g = calc_FIM(q, r, xi, flux, model)
        fisher_error = 1 / np.sqrt(np.diag(g)) #Calculate the Fisher errors.

        fit_errors.append(fit_error)
        fisher_errors.append(fisher_error)

    #Plot of log fit errors against log time constants.
    fit_errors = np.array(fit_errors)
    fit_fig    = plt.figure(num=1, dpi=600)
    fit_ax     = fit_fig.add_subplot(111)
    fit_ax.set_xlabel("Log Time Constant")
    fit_ax.set_ylabel("Log L-BFGS-B Error")

    #Plot of log Fisher errors against log time constants.
    fisher_errors = np.array(fisher_errors)
    fisher_fig    = plt.figure(num=2, dpi=600)
    fisher_ax     = fisher_fig.add_subplot(111)
    fisher_ax.set_xlabel("Log Time Constant")
    fisher_ax.set_ylabel("Log Fisher Error")

    #Plot the relationship of error against noise for each parameter.
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
