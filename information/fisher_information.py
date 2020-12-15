import os, sys
sys.path.append("../simulation") #Adds higher directory to python modules path.

import numpy as np
import matplotlib.pyplot as plt

from refnx.dataset  import ReflectDataset
from refnx.analysis import Objective, CurveFitter

from simulate import simulate_single_contrast, vary_model

def calc_FIM(q, xi, flux, model):
    """Calculates the Fisher information matrix for given a dataset and model.

    Args:
        q (numpy.ndarray): array of Q values.
        xi (list): list of refnx Parameter objects representing each varying parameter value.
        flux (numpy.ndarray): array of flux values corresponding to each Q value.
        model (refnx.reflect.ReflectModel): the model to calculate the gradient with.

    Returns:
        numpy.ndarray: Fisher information matrix.

    """
    n = len(q)
    m = len(xi)
    J = np.zeros((n,m))
    #Calculate the gradient of the model reflectivity with every model parameter for every model data point.
    for i in range(n):
        for j in range(m):
            J[i,j] = gradient(model, xi[j], q[i])

    r = model(q) #Use model reflectivity values
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
        float: the two-point gradient.

    """
    step = parameter.value * step #0.5% step by default
    old = parameter.value

    x1 = parameter.value = old - step #First point
    y1 = model(q_point) #Get new r value with altered model.

    x2 = parameter.value = old + step #Second point
    y2 = model(q_point)

    parameter.value = old #Reset parameter
    return (y2-y1) / (x2-x1) #Return the gradient

def compare_fit_variance(structure, angle_times, save_path, n=1000):
    """Compares the multiplicative inverse Fisher information (variance) with
       variance in parameter estimation using traditional fitting for `n` fits.

    Args:
        structure (refnx.reflect.Structure): the structure to simulate the experiment for.
        angle_times (dict): dictionary of number of points and measurement times to use for each angle.
        save_path (string): path to directory to save Fisher and fit variances.
        n (int): number of fits to run.

    """
    param_estimates = []
    for i in range(n): #Fit `n` times.
        if i % 10 == 0: #Display progress every 10 fits.
            print("{0}/{1}...".format(i, n))

        #Simulate the experiment using the given angle, number of points and time.
        model, data = simulate_single_contrast(*structure(), angle_times)
        vary_model(model) #Vary the SLD and thickness of each layer and set them to random values.

        q, r, r_error, flux = data[:,0], data[:,1], data[:,2], data[:,3]

        #Fit the model using differential evolution.
        objective = Objective(model, ReflectDataset([q, r, r_error]))
        fitter = CurveFitter(objective)
        fitter.fit('differential_evolution', verbose=False)

        xi = objective.varying_parameters() #Get the parameter estimates.
        param_estimates.append([param.value for param in xi])

    #Calculate the variances in parameter estimates from `n` fits.
    param_vars = np.var(np.array(param_estimates), axis=0)

    #Calculate the Fisher information matrix from the ground truth model.
    model, data = simulate_single_contrast(*structure(), angle_times)
    q, flux = data[:,0], data[:,3]
    xi = []    
    for component in model.structure.components[1:-1]:
        xi.append(component.sld.real)
        xi.append(component.thick)
    
    g = calc_FIM(q, xi, flux, model) #Calculate the Fisher information matrix.
    inv_fisher = 1 / np.diag(g)

    print("Variance in Parameter Estimation:", param_vars)
    print("Inverse Fisher Information:", inv_fisher)

    #Save the results to a .txt file.
    with open(save_path+"/variance_comparison.txt", "w") as file:
        file.write("Variance in Parameter Estimation:\n")
        file.write(str(param_vars)+"\n"*2)
        file.write("Inverse Fisher Information:\n")
        file.write(str(inv_fisher))

def compare_errors(structure, angle_times, times, save_path):
    """Compares traditional fitting errors and Fisher errors with increasing time.

    Args:
        structure (refnx.reflect.Structure): the structure to simulate the experiment for.
        angle_times (dict): dictionary of number of points to use for each angle.
        times (numpy.ndarray): array of time constants to use for the comparison.
        save_path(string): path to directory for saving figures.

    """
    fit_errors, fisher_errors = [], []
    for i, time in enumerate(times, 1):
        print(">>> Fitting {0}/{1}...".format(i, len(times))) #Display progress

        #Use the time from the times array but keep the number of points.
        for angle in angle_times:
            angle_times[angle] = (angle_times[angle][0], time)

        #Simulate the experiment using the given angle, number of points and time.
        model, data = simulate_single_contrast(*structure(), angle_times)
        vary_model(model) #Vary the SLD and thickness of each layer and set them to random values.

        q, r, r_error, flux = data[:,0], data[:,1], data[:,2], data[:,3]

        #Fit the data using differential evolution.
        objective = Objective(model, ReflectDataset([q, r, r_error]))
        fitter = CurveFitter(objective)
        fitter.fit('differential_evolution', verbose=False)

        xi = objective.varying_parameters()
        fit_error = [param.stderr for param in xi] #Get the fitting errors.
        g = calc_FIM(q, xi, flux, model)
        fisher_error = 1 / np.sqrt(np.diag(g)) #Calculate the Fisher errors.

        fit_errors.append(fit_error)
        fisher_errors.append(fisher_error)

    #Plot of log fit errors against log time constants.
    fit_errors = np.array(fit_errors)
    fit_fig    = plt.figure(num=1, dpi=600)
    fit_ax     = fit_fig.add_subplot(111)
    fit_ax.set_xlabel("Log Time Constant")
    fit_ax.set_ylabel("Log Fitting Error")

    #Plot of log Fisher errors against log time constants.
    fisher_errors = np.array(fisher_errors)
    fisher_fig    = plt.figure(num=2, dpi=600)
    fisher_ax     = fisher_fig.add_subplot(111)
    fisher_ax.set_xlabel("Log Time Constant")
    fisher_ax.set_ylabel("Log Fisher Error")

    #Plot the relationship of error against noise for each parameter.
    for i in range(len(xi)):
        param = xi[i].name
        fit_ax.plot(np.log(times), np.log(fit_errors[:,i]), label=param)
        fisher_ax.plot(np.log(times), np.log(fisher_errors[:,i]), label=param)

    fit_ax.legend()
    fisher_ax.legend()
    fit_fig.savefig(save_path+"/fit_errors_with_time.png", dpi=600)
    fisher_fig.savefig(save_path+"/fisher_errors_with_time.png", dpi=600)

if __name__ == "__main__":
    from structures import similar_sld_sample_1, similar_sld_sample_2
    from structures import thin_layer_sample_1,  thin_layer_sample_2
    from structures import easy_sample, many_param_sample

    structure   = easy_sample #Choose structure here.
    angle_times = {0.7: (70, 5), #Angle: (Points, Time)
                   2.0: (70, 20)}

    save_path = "./results/"+structure.__name__
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    times = 10**np.arange(0.5, 5, 0.2, dtype=float)
    #compare_errors(structure, angle_times, times, save_path)

    compare_fit_variance(structure, angle_times, save_path, n=200)
