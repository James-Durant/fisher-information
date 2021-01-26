import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.linear_model import LinearRegression

from refnx.dataset  import ReflectDataset
from refnx.analysis import Objective, CurveFitter

from simulation.simulate import simulate_single_contrast, vary_model
from information.utils   import calc_FIM

def compare_errors(structure, angle_points, times, save_path):
    """Compares traditional fitting errors and FIM errors with increasing time.

    Args:
        structure (function): the structure to simulate the experiment on.
        angle_points (dict): dictionary of points to use for each angle.
        times (numpy.ndarray): array of times to use for the comparison.
        save_path(string): path to directory to save figures to.

    """
    fit_errors, FIM_errors = [], []
    for i, time in enumerate(times, 1):
        #Calculate the fit and FIM errors for 10 simulations using the same measurement time.
        fit_errors_time, FIM_errors_time = [], []
        for _ in range(10):
            #Create dictionary of number of points and measurement time for each angle.
            angle_times = {angle: (angle_points[angle], time) for angle in angle_points}
            model, data = simulate_single_contrast(structure(), angle_times) #Simulate the experiment.
            vary_model(model) #Vary the SLD and thickness of each layer and set them to random values.

            q, r, r_error, counts = data[:,0], data[:,1], data[:,2], data[:,3]

            #Fit the data using differential evolution.
            objective = Objective(model, ReflectDataset([q, r, r_error]))
            fitter = CurveFitter(objective)
            fitter.fit('differential_evolution', verbose=False)

            xi = objective.varying_parameters()
            fit_error = [param.stderr for param in xi] #Get the fitting errors.
            g = calc_FIM(q, xi, counts, model)
            FIM_error = 1 / np.sqrt(np.diag(g)) #Calculate the FIM errors.

            fit_errors_time.append(fit_error)
            FIM_errors_time.append(FIM_error)

        #Use the mean of the 10 errors for the time point on the plot.
        fit_errors.append(np.mean(fit_errors_time, axis=0))
        FIM_errors.append(np.mean(FIM_errors_time, axis=0))
        print(">>> Fitting {0}/{1}...".format(i*10, len(times)*10)) #Display progress

    #Plot the mean fit and FIM errors vs. increasing time.
    names = [param.name for param in xi]
    plot_errors(times, np.array(fit_errors), np.array(FIM_errors), names)

def plot_errors(times, fit_errors, FIM_errors, names):
    """Plots log time against log fitting error and log FIM error for each parameter
       and performs linear regression to calculate the gradient in each case.

    Args:
        times (numpy.ndarray): array of times for which the errors were calculated using.
        fit_errors (numpy.ndarray): parameter fitting errors for each time.
        FIM_errors (numpy.ndarray): FIM parameter errors for each time.
        names (list): list of string parameter names.

    """
    #Create the plot of log fit errors against log time.
    fit_fig = plt.figure(figsize=[7,5], num=1, dpi=600)
    fit_ax  = fit_fig.add_subplot(111)
    fit_ax.set_xlabel("Log Time",          weight='bold')
    fit_ax.set_ylabel("Log Fitting Error", weight='bold')

    #Create the plot of log FIM errors against log time.
    FIM_fig = plt.figure(figsize=[7,5], num=2, dpi=600)
    FIM_ax  = FIM_fig.add_subplot(111)
    FIM_ax.set_xlabel("Log Time",      weight='bold')
    FIM_ax.set_ylabel("Log FIM Error", weight='bold')

    log_time = np.log(times).reshape(-1,1) #Reshape for linear regression.
    #Plot the relationship of fitting and FIM error against time for each parameter.
    for i in range(len(names)):
        #Take the log of the fitting and FIM errors and reshape for linear regression.
        log_fit = np.log(fit_errors[:,i]).reshape(-1,1)
        log_FIM = np.log(FIM_errors[:,i]).reshape(-1,1)

        #Create linear regressors to calculate the gradient of lines.
        fit_reg = LinearRegression().fit(log_time, log_fit)
        FIM_reg = LinearRegression().fit(log_time, log_FIM)

        #Round the gradients to 3dp.
        fit_m = np.format_float_positional(fit_reg.coef_[0][0], precision=3, unique=False, trim='k')
        FIM_m = np.format_float_positional(FIM_reg.coef_[0][0], precision=3, unique=False, trim='k')

        #Plot log time against log error for the parameter.
        fit_ax.plot(log_time, log_fit, label=names[i]+", m="+fit_m)
        FIM_ax.plot(log_time, log_FIM, label=names[i]+", m="+FIM_m)

    fit_ax.legend()
    FIM_ax.legend()
    fit_fig.savefig(save_path+"/fit_errors_with_time.png", dpi=600)
    FIM_fig.savefig(save_path+"/FIM_errors_with_time.png", dpi=600)

if __name__ == "__main__":
    from simulation.structures import similar_sld_sample_1, similar_sld_sample_2
    from simulation.structures import thin_layer_sample_1,  thin_layer_sample_2
    from simulation.structures import easy_sample, many_param_sample

    structure = easy_sample #Choose structure here.
    save_path = "./results/"+structure.__name__
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    angle_points = {0.7: 70, #Angle: Points
                    2.0: 70}
    times = 10**np.arange(0.5, 5.5, 0.1, dtype=float)
    compare_errors(structure, angle_points, times, save_path)
