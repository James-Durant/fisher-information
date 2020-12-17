import os, sys
sys.path.append("../simulation") #Adds directory to Python modules path.

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from refnx.dataset  import ReflectDataset
from refnx.analysis import Objective, CurveFitter

from fisher   import calc_FIM
from simulate import simulate_single_contrast, vary_model

def compare_errors(structure, angle_points, times, save_path):
    """Compares traditional fitting errors and Fisher errors with increasing time.

    Args:
        structure (refnx.reflect.Structure): the structure to simulate the experiment for.
        angle_points (dict): dictionary of number of points to use for each angle.
        times (numpy.ndarray): array of time constants to use for the comparison.
        save_path(string): path to directory for saving figures.

    """
    fit_errors, fisher_errors = [], []
    for i, time in enumerate(times, 1):
        print(">>> Fitting {0}/{1}...".format(i, len(times))) #Display progress

        #Create dictionary of number of points and measurement time for each angle.
        angle_times = {angle: (angle_points[angle], time) for angle in angle_points}
        model, data = simulate_single_contrast(*structure(), angle_times) #Simulate the experiment.
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
        
    names = [param.name for param in xi]
    plot_errors(times, np.array(fit_errors), np.array(fisher_errors), names)

def plot_errors(times, fit_errors, fisher_errors, names):
    """Plots log time against log fitting error and log Fisher error for each parameter
       and performs linear regression to caluclate the gradient in each case.

    Args:
        times (numpy.ndarray): array of times for which the errors were calculated using.
        fit_errors (numpy.ndarray): parameter fitting errors for each time.
        fisher_errors (numpy.ndarray): parameter Fisher errors for each time.
        names (list): list of string parameter names.

    """
    #Create the plot of log fit errors against log time.
    fit_fig    = plt.figure(figsize=[7,5], num=1, dpi=600)
    fit_ax     = fit_fig.add_subplot(111)
    fit_ax.set_xlabel("Log Time",          weight='bold')
    fit_ax.set_ylabel("Log Fitting Error", weight='bold')

    #Create the plot of log Fisher errors against log time.
    fisher_fig    = plt.figure(figsize=[7,5], num=2, dpi=600)
    fisher_ax     = fisher_fig.add_subplot(111)
    fisher_ax.set_xlabel("Log Time",         weight='bold')
    fisher_ax.set_ylabel("Log Fisher Error", weight='bold')

    log_time = np.log(times).reshape(-1,1) #Reshape for sklearn LinearRegression.
    #Plot the relationship of fitting and Fisher error against time for each parameter.
    for i in range(len(names)):
        #Take the log of the fitting and Fisher errors and reshape for linear regression.
        log_fit    = np.log(fit_errors[:,i]).reshape(-1,1)
        log_fisher = np.log(fisher_errors[:,i]).reshape(-1,1)
        
        #Create linear regressors to calculate the gradient of lines.
        fit_reg    = LinearRegression().fit(log_time, log_fit)
        fisher_reg = LinearRegression().fit(log_time, log_fisher)
        
        #Round the gradients to 3dp.
        fit_m    = np.format_float_positional(fit_reg.coef_[0][0],    precision=3, unique=False, trim='k')
        fisher_m = np.format_float_positional(fisher_reg.coef_[0][0], precision=3, unique=False, trim='k')
        
        #Plot log time against log error for the parameter.
        fit_ax.plot(log_time,    log_fit,    label=names[i]+", m="+fit_m)
        fisher_ax.plot(log_time, log_fisher, label=names[i]+", m="+fisher_m)

    fit_ax.legend()
    fisher_ax.legend()
    fit_fig.savefig(save_path+"/fit_errors_with_time.png",       dpi=600)
    fisher_fig.savefig(save_path+"/fisher_errors_with_time.png", dpi=600)

if __name__ == "__main__":
    from structures import similar_sld_sample_1, similar_sld_sample_2
    from structures import thin_layer_sample_1,  thin_layer_sample_2
    from structures import easy_sample, many_param_sample

    structure = easy_sample #Choose structure here.
    save_path = "./results/"+structure.__name__
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    angle_points = {0.7: 70, #Angle: Points
                    2.0: 70}
    times = 10**np.arange(0.5, 5.5, 0.1, dtype=float)
    compare_errors(structure, angle_points, times, save_path)
