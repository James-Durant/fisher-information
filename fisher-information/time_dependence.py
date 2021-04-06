import matplotlib.pyplot as plt
import numpy as np
import os

from typing import List, Callable
from numpy.typing import ArrayLike

from sklearn.linear_model import LinearRegression

from refnx.dataset import ReflectDataset
from refnx.reflect import ReflectModel
from refnx.analysis import Parameter, Objective, CurveFitter

from simulate import simulate_single_contrast, AngleTimes
from structures import QCS_sample

from utils import vary_structure, fisher_single_contrast
from plotting import save_plot

def simulated_projections(structure: Callable, angle_times: AngleTimes,
                          multipliers: ArrayLike, save_path: str) -> None:
    """Compares the predicted (using the uncertainty inversely proportional to
       time squared relationship) and actual parameter uncertainties vs.
       increasing measurement time for simulated data of a given structure.

    Args:
        structure (function): structure to simulate the experiments on.
        angle_times (dict): initial points and times for each angle.
        multipliers (numpy.ndarray): array of time multipliers.
        save_path (str): path to directory to save figures to.

    """
    save_path = os.path.join(save_path, structure.__name__)

    # Iterate over each given time multiplier.
    errors = []
    for i, factor in enumerate(multipliers, 1):
        # Multiply the initial times by the current multiplier for each angle.
        new_angle_times = {angle: (angle_times[angle][0],
                                   angle_times[angle][1]*factor)
                           for angle in angle_times}

        # Simulate the experiment and fit.
        objective = Objective(*simulate_single_contrast(vary_structure(structure()),
                                                        new_angle_times))
        CurveFitter(objective).fit('differential_evolution', verbose=False)

        # Record the parameter uncertainties with this time multiplier.
        errors.append([param.stderr for param in objective.varying_parameters()])

        # Display progress.
        print('>>> {0}/{1}'.format(i, len(multipliers)))

    # Plot the uncertainty projections and actual uncertainties vs. time.
    errors = np.asarray(errors)
    xi = objective.varying_parameters()
    plot_projections(multipliers, errors, xi, save_path)

def measured_projections(data_path: str, save_path: str, scale: float=1,
                         bkg: float=8e-7, dq: float=2.5) -> None:
    """Compares the predicted (using the uncertainty inversely proportional to
       time squared relationship) and actual parameter uncertainties vs.
       increasing measurement time for measured data of the QCS sample.

    Args:
        data_path (str): path to directory containing measured QCS data.
        save_path (str): path to directory to save figures to.
        scale (float): model experimental scale factor.
        bkg (float): model background parameter.
        dq (float): model instrument resolution.

    """
    data_path = os.path.join(data_path, 'QCS_sample', 'time_data')
    save_path = os.path.join(save_path, 'QCS_sample')

    # Load the measured datasets corresponding to different measurement times.
    datasets = [np.loadtxt(os.path.join(data_path,'{}.dat'.format(i+1)), delimiter=',')
                for i in range(10)]
    errors = []

    # Iterate over each dataset.
    for i in range(1, 11):
        points = 96*i
        data = np.zeros((points, 3))
        q, r, r_error = [], [], []
        for sliced in datasets[:i]:
            q += sliced[:,0].tolist()
            r += sliced[:,1].tolist()
            r_error += sliced[:,2].tolist()

        data[:,0] = q
        data[:,1] = r
        data[:,2] = r_error

        data = data[(data != 0).all(1)]  #Remove points of zero reflectivity.
        data = data[data[:,0].argsort()] #Sort by Q

        # Load data into refnx format and scale.
        refdata = ReflectDataset([data[:,0], data[:,1], data[:,2]])
        refdata.scale(np.max(refdata.y))

        # Fit the data and record the parameter errors.
        model = ReflectModel(vary_structure(QCS_sample()),
                             scale=scale, bkg=bkg, dq=dq)
        objective = Objective(model, refdata)
        CurveFitter(objective).fit('differential_evolution', verbose=False)

        xi = objective.varying_parameters()
        errors.append([param.stderr for param in xi])

    # Plot the uncertainty projections and actual uncertainties vs. time.
    errors = np.asarray(errors)
    time_factors = np.arange(1, 11, 1)
    plot_projections(time_factors, errors, xi, save_path)

def plot_projections(multipliers: ArrayLike, errors: ArrayLike,
                     xi: List[Parameter], save_path: str) -> None:
    """Plots uncertainty projections (using initial errors) and actual
       uncertainties vs. measurement time.

    Args:
        multipliers (numpy.ndarray): array of time multipliers.
        errors (numpy.ndarray): array of actual errors from fitting.
        xi (list): parameters that were varied when fitting.
        save_path (str): path to directory to save figures to.

    """
    fig = plt.figure(figsize=[7,5], dpi=600)
    ax = fig.add_subplot(111)

    # Iterate over each parameter.
    for i, param in enumerate(xi):
        # Plot the actual errors vs. time for the parameter.
        ax.plot(multipliers, errors[:,i], label=param.name,
                linestyle='', marker='x', zorder=2)

        # Plot the projected errors vs. time for the parameter.
        # The projection is taken using the initial errors.
        times = np.arange(multipliers [0], multipliers [-1], 0.01)
        ax.plot(times, errors[0][i]/np.sqrt(times),
                color='black', lw=0.8, zorder=1)

    ax.set_xlabel('Time')
    ax.set_ylabel('Parameter Uncertainty')
    ax.legend()
    save_plot(fig, save_path, 'error_projection') # Save the plot.

def compare_errors(structure: Callable, angle_times: AngleTimes,
                   multipliers: ArrayLike, save_path: str) -> None:
    """Compares traditional fitting errors and FIM errors with increasing time.

    Args:
        structure (function): structure to simulate the experiment on.
        initial_angle_times (dict): initial points and times for each angle.
        multipliers (numpy.ndarray): array of time multipliers.
        save_path (str): path to directory to save figures to.

    """
    save_path = os.path.join(save_path, structure.__name__)

    # Iterate over the given times.
    fit_errors, FIM_errors = [], []
    for i, multiplier in enumerate(multipliers, 1):
        # Calculate the fit and FIM errors for 10 simulations using
        # the same measurement time.
        fit_errors_time, FIM_errors_time = [], []

        for _ in range(10):
            # Number of points and measurement times for each angle.
            new_angle_times = {angle: (angle_times[angle][0],
                                       angle_times[angle][1]*multiplier)
                               for angle in angle_times}

            # Simulate the experiment.
            model, data, counts = simulate_single_contrast(structure(),
                                                           new_angle_times,
                                                           include_counts=True)

            # Vary layers' SLDs and thicknesses and set them to random values.
            vary_structure(model.structure, random_init=True)

            # Fit the data using differential evolution.
            objective = Objective(model, data)
            CurveFitter(objective).fit('differential_evolution', verbose=False)

            # Get the fitting errors.
            xi = objective.varying_parameters()
            fit_errors_time.append([param.stderr for param in xi])

            # Calculate the FIM errors.
            g = fisher_single_contrast(data.x, xi, counts, model)
            FIM_errors_time.append(1 / np.sqrt(np.diag(g)))

        # Use the mean of the 10 errors for the time point.
        fit_errors.append(np.mean(fit_errors_time, axis=0))
        FIM_errors.append(np.mean(FIM_errors_time, axis=0))

        # Display progress.
        print('>>> {0}/{1}'.format(i*10, len(multipliers)*10))

    # Plot the mean fit and FIM errors vs. increasing time.
    names = [param.name for param in xi]
    fit_errors = np.asarray(fit_errors)
    FIM_errors = np.asarray(FIM_errors)
    plot_errors(multipliers, fit_errors, FIM_errors, names, save_path)

def plot_errors(multipliers: ArrayLike, fit_errors: ArrayLike,
                FIM_errors: ArrayLike, names: List[str], save_path: str) -> None:
    """Plots log fitting error and log FIM error vs. log time multiplier for
       each parameter and performs linear regression for gradient in each case.

    Args:
        multipliers (numpy.ndarray): multipliers for which errors were calculated.
        fit_errors (numpy.ndarray): parameter fitting errors for each time.
        FIM_errors (numpy.ndarray): FIM parameter errors for each time.
        names (list): parameter names.
        save_path (str): path to directory to save figures to.

    """
    # Create the plot of log fit errors against log time multipliers.
    fit_fig = plt.figure(figsize=[7,5], num=1, dpi=600)
    fit_ax = fit_fig.add_subplot(111)
    fit_ax.set_xlabel('Log Time Multiplier', weight='bold')
    fit_ax.set_ylabel('Log Fitting Error', weight='bold')

    # Create the plot of log FIM errors against log time multipliers.
    FIM_fig = plt.figure(figsize=[7,5], num=2, dpi=600)
    FIM_ax = FIM_fig.add_subplot(111)
    FIM_ax.set_xlabel('Log Time Multiplier', weight='bold')
    FIM_ax.set_ylabel('Log FIM Error', weight='bold')

    # Reshape for linear regression.
    log_time = np.log(multipliers).reshape(-1, 1)

    # Iterate over each parameter.
    for i in range(len(names)):
        # log of the fitting and FIM errors and reshape for linear regression.
        log_fit = np.log(fit_errors[:,i]).reshape(-1,1)
        log_FIM = np.log(FIM_errors[:,i]).reshape(-1,1)

        # Create linear regressors to calculate the gradient of lines.
        fit_reg = LinearRegression().fit(log_time, log_fit)
        FIM_reg = LinearRegression().fit(log_time, log_FIM)

        # Round the gradients to 3dp.
        fit_m = np.format_float_positional(fit_reg.coef_[0][0], precision=3,
                                           unique=False, trim='k')

        FIM_m = np.format_float_positional(FIM_reg.coef_[0][0], precision=3,
                                           unique=False, trim='k')

        # Plot log time multipliers against log error for the parameter.
        fit_ax.plot(log_time, log_fit, label=names[i]+', m='+fit_m)
        FIM_ax.plot(log_time, log_FIM, label=names[i]+', m='+FIM_m)

    fit_ax.legend()
    FIM_ax.legend()

    save_plot(fit_fig, save_path, 'fit_errors_with_time') # Save the plots.
    save_plot(FIM_fig, save_path, 'FIM_errors_with_time')

if __name__ == '__main__':
    from structures import similar_sld_sample_1, similar_sld_sample_2
    from structures import thin_layer_sample_1, thin_layer_sample_2
    from structures import easy_sample, many_param_sample

    save_path = './results'

    structure = easy_sample # Choose structure here.
    angle_times = {0.7: (70, 1), # Angle: (Points, Time)
                   2.0: (70, 4)}

    # Investigates how FIM and fitting errors change with measurement time.
    multipliers = 10**np.arange(0.5, 5.5, 2, dtype=float)
    compare_errors(structure, angle_times, multipliers, save_path)

    # Investigates how the uncertainty inversely proportional to time squared
    # relationship holds in practice.
    #reduction_range = np.arange(1, 10, 0.25)
    #simulated_projections(structure, angle_times, reduction_range, save_path)

    data_path = './data'
    #measured_projections(data_path, save_path)
