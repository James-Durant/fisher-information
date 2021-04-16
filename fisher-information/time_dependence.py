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
    """Compares the predicted and actual parameter uncertainties vs.
       increasing measurement time for simulated data of a given structure
       (using the uncertainty inversely proportional to time squared
       relationship).

    Args:
        structure (function): structure to simulate the experiments on.
        angle_times (dict): initial points and times for each angle.
        multipliers (numpy.ndarray): array of time multipliers.
        save_path (str): path to directory to save figures to.

    """
    save_path = os.path.join(save_path, structure.__name__)

    # Iterate over each given time multiplier.
    uncertainties = []
    for i, factor in enumerate(multipliers, 1):
        # Multiply the initial times by the current multiplier for each angle.
        new_angle_times = {angle: (angle_times[angle][0],
                                   angle_times[angle][1]*factor)
                           for angle in angle_times}

        # Simulate the experiment and fit.
        model, data = simulate_single_contrast(vary_structure(structure()),
                                               new_angle_times)
        objective = Objective(model, data)
        CurveFitter(objective).fit('differential_evolution', verbose=False)

        # Record the parameter uncertainties with this time multiplier.
        uncertainties.append([param.stderr
                              for param in objective.varying_parameters()])

        # Display progress.
        print('>>> {0}/{1}'.format(i, len(multipliers)))

    # Plot the uncertainty projections and actual uncertainties vs. time.
    uncertainties = np.asarray(uncertainties)
    xi = objective.varying_parameters()
    plot_projections(multipliers, uncertainties, xi, 'simulated', save_path)

def measured_projections(data_path: str, save_path: str, scale: float=1,
                         bkg: float=8e-7, dq: float=2.5) -> None:
    """Compares the predicted and actual parameter uncertainties vs.
       increasing measurement time for measured data of the QCS sample.
       (using the uncertainty inversely proportional to time squared
       relationship).

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
    datasets = [np.loadtxt(os.path.join(data_path, '{}.dat'.format(i+1)),
                           delimiter=',')
                for i in range(10)]
    uncertainties = []

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

        data = data[(data != 0).all(1)] # Remove points of zero reflectivity.
        data = data[data[:,0].argsort()] # Sort by Q.

        # Load data into refnx format and scale.
        refdata = ReflectDataset([data[:,0], data[:,1], data[:,2]])
        refdata.scale(np.max(refdata.y))

        # Fit the data and record the parameter uncertainties.
        model = ReflectModel(vary_structure(QCS_sample()),
                             scale=scale, bkg=bkg, dq=dq)
        objective = Objective(model, refdata)
        CurveFitter(objective).fit('differential_evolution', verbose=False)

        xi = objective.varying_parameters()
        uncertainties.append([param.stderr for param in xi])

    # Plot the uncertainty projections and actual uncertainties vs. time.
    uncertainties = np.asarray(uncertainties)
    time_factors = np.arange(1, 11, 1)
    plot_projections(time_factors, uncertainties, xi, 'measured', save_path)

def plot_projections(multipliers: ArrayLike, uncertainties: ArrayLike,
                     xi: List[Parameter], source: str, save_path: str) -> None:
    """Plots uncertainty projections (using initial uncertainties) and fitting
       uncertainties vs. measurement time.

    Args:
        multipliers (numpy.ndarray): array of time multipliers.
        uncertainties (numpy.ndarray): array of uncertainties from fitting.
        xi (list): parameters that were varied when fitting.
        source (str): either 'simulated' or 'measured'.
        save_path (str): path to directory to save figures to.

    """
    fig = plt.figure(figsize=[9,7], dpi=600)
    ax = fig.add_subplot(111)

    colours = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Iterate over each parameter.
    for i, param in enumerate(xi):
        # Plot the fitting uncertainty vs. time for the parameter.
        ax.plot(multipliers, uncertainties[:,i], label=param.name+' Actual',
                color=colours[i], linestyle='', marker='x', zorder=2)

        # Plot the projected uncertainty vs. time for the parameter.
        # The projection is taken using the initial uncertainties.
        times = np.arange(multipliers [0], multipliers [-1], 0.01)
        ax.plot(times, uncertainties[0][i]/np.sqrt(times),
                label=param.name+' Projected', color=colours[i],
                lw=0.8, zorder=1)

    ax.set_ylabel('Parameter Uncertainty', weight='bold')
    ax.legend(loc='upper right')

    if source == 'simulated':
        ax.set_xlabel('Time Multipler', fontsize=11, weight='bold')
        save_plot(fig, save_path, 'uncertainty_projection_simulated')
    else:
        ax.set_xlabel('Time', fontsize=11, weight='bold')
        save_plot(fig, save_path, 'uncertainty_projection_measured')

def compare_uncertainties(structure: Callable, angle_times: AngleTimes,
                          multipliers: ArrayLike, save_path: str) -> None:
    """Compares fitting uncertainties and FIM uncertainties with
       increasing time.

    Args:
        structure (function): structure to simulate the experiment on.
        initial_angle_times (dict): initial points and times for each angle.
        multipliers (numpy.ndarray): array of time multipliers.
        save_path (str): path to directory to save figures to.

    """
    save_path = os.path.join(save_path, structure.__name__)

    # Iterate over the given times.
    fit_uncertainties, FIM_uncertainties, errors = [], [], []
    for i, multiplier in enumerate(multipliers, 1):
        # Calculate the fit and FIM uncertainties for 10 simulations using
        # the same measurement time.
        fit_uncertainties_time, FIM_uncertainties_time = [], []
        fit_values = []

        for _ in range(10):
            # Number of points and measurement times for each angle.
            new_angle_times = {angle: (angle_times[angle][0],
                                       angle_times[angle][1]*multiplier)
                               for angle in angle_times}

            # Simulate the experiment.
            simulated = simulate_single_contrast(vary_structure(structure()),
                                                 new_angle_times,
                                                 include_counts=True)
            model, data, counts = simulated

            # Fit the data using differential evolution.
            objective = Objective(model, data)
            xi = objective.varying_parameters()

            # Get the ground truth parameter values.
            true = np.asarray([param.value for param in xi])

            # Calculate the FIM matrix and retrieve the FIM uncertainties.
            g = fisher_single_contrast(data.x, xi, counts, model)
            FIM_uncertainties_time.append(1 / np.sqrt(np.diag(g)))

            # Fit using differential evolution and record the uncertainties.
            CurveFitter(objective).fit('differential_evolution', verbose=False)
            fit_uncertainties_time.append([param.stderr for param in xi])

            # Record the fitted parameter values.
            fit_values.append([param.value for param in xi])

        # Use the mean of the 10 uncertainties for the time point.
        fit_uncertainties.append(np.mean(fit_uncertainties_time, axis=0))
        FIM_uncertainties.append(np.mean(FIM_uncertainties_time, axis=0))

        # Calculate the mean absolute error in the 10 fits.
        errors.append(np.mean(abs(np.array(fit_values) - true), axis=0))

        # Display progress.
        print('>>> {0}/{1}'.format(i*10, len(multipliers)*10))

    names = [param.name for param in xi]

    # Plot the mean fit and FIM uncertainties vs. increasing time.
    plot_uncertainties(multipliers, np.array(fit_uncertainties),
                       np.array(FIM_uncertainties), names, save_path)

    # Plot the mean absolute error vs. increasing time.
    plot_fitting_error(multipliers, np.asarray(errors), names, save_path)

def plot_uncertainties(multipliers: ArrayLike, fit_uncertainties: ArrayLike,
                       FIM_uncertainties: ArrayLike, names: List[str],
                       save_path: str) -> None:
    """Plots log fitting uncertainties and log FIM uncertainties vs. log time
       multiplier for each parameter and performs linear regression for an
       approximation of the gradient in each case.

    Args:
        multipliers (numpy.ndarray): time multipliers.
        fit_uncertainties (numpy.ndarray): fitting uncertainties for each time.
        FIM_uncertainties (numpy.ndarray): FIM uncertainties for each time.
        names (list): parameter names.
        save_path (str): path to directory to save figures to.

    """
    # Create the plot of log fitting uncertainties vs. log time multipliers.
    fit_fig = plt.figure(figsize=[7,5], num=1, dpi=600)
    fit_ax = fit_fig.add_subplot(111)
    fit_ax.set_xlabel('Log Time Multiplier', weight='bold')
    fit_ax.set_ylabel('Log Fitting Uncertainty', weight='bold')

    # Create the plot of log FIM uncertainties vs. log time multipliers.
    FIM_fig = plt.figure(figsize=[7,5], num=2, dpi=600)
    FIM_ax = FIM_fig.add_subplot(111)
    FIM_ax.set_xlabel('Log Time Multiplier', weight='bold')
    FIM_ax.set_ylabel('Log FIM Uncertainty', weight='bold')

    # Reshape for linear regression.
    log_time = np.log(multipliers).reshape(-1, 1)

    # Iterate over each parameter.
    for i in range(len(names)):
        # log of the fitting and FIM uncertainties and reshape for regression.
        log_fit = np.log(fit_uncertainties[:,i]).reshape(-1,1)
        log_FIM = np.log(FIM_uncertainties[:,i]).reshape(-1,1)

        # Create linear regressors to calculate the gradient of lines.
        fit_reg = LinearRegression().fit(log_time, log_fit)
        FIM_reg = LinearRegression().fit(log_time, log_FIM)

        # Round the gradients to 3dp.
        fit_m = np.format_float_positional(fit_reg.coef_[0][0], precision=3,
                                           unique=False, trim='k')

        FIM_m = np.format_float_positional(FIM_reg.coef_[0][0], precision=3,
                                           unique=False, trim='k')

        # Plot log time multipliers against log uncertainty for the parameter.
        fit_ax.plot(log_time, log_fit, label=names[i]+', m='+fit_m)
        FIM_ax.plot(log_time, log_FIM, label=names[i]+', m='+FIM_m)

    fit_ax.legend(loc='upper right')
    FIM_ax.legend(loc='upper right')

    save_plot(fit_fig, save_path, 'fitting_uncertainties_vs_time')
    save_plot(FIM_fig, save_path, 'FIM_uncertainties_vs_time')

def plot_fitting_error(multipliers: ArrayLike, errors: ArrayLike,
                       names: List[str], save_path: str) -> None:
    """Plots mean absolute error vs. log time multiplier for each parameter.

    Args:
        multipliers (numpy.ndarray): time multipliers.
        errors (numpy.ndarray): mean absolute errors for each multiplier.
        names (list): parameter names.
        save_path (str): path to directory to save figures to.

    """
    # Create the plot of log mean absolute error vs. log time multipliers.
    fig = plt.figure(figsize=[7,5], dpi=600)
    ax = fig.add_subplot(111)

    log_time = np.log(multipliers)

    # Iterate over each parameter and plot the MAE vs. time.
    for i in range(len(names)):
        ax.plot(log_time, errors[:,i], label=names[i])

    ax.set_xlabel('Log Time Multiplier', weight='bold')
    ax.set_ylabel('Mean Absolute Error', weight='bold')
    ax.legend(loc='upper right')

    save_plot(fig, save_path, 'fitting_error_vs_time') # Save the plot.

if __name__ == '__main__':
    from structures import similar_sld_sample_1, similar_sld_sample_2
    from structures import thin_layer_sample_1, thin_layer_sample_2
    from structures import easy_sample, QCS_sample, many_param_sample

    save_path = './results'

    structure = easy_sample # Choose structure here.
    angle_times = {0.7: (70, 5), # Angle: (Points, Time)
                   2.0: (70, 20)}

    # Investigates how FIM and fitting uncertainties change with time.
    multipliers = 10**np.arange(0, 3, 0.01, dtype=float)
    compare_uncertainties(structure, angle_times, multipliers, save_path)

    # Investigates how the uncertainty inversely proportional to time squared
    # relationship holds in practice.
    multipliers = 10**np.arange(0, 2, 0.05, dtype=float)
    simulated_projections(structure, angle_times, multipliers, save_path)

    # Plot the relationship for measured data of the QCS sample.
    data_path = './data'
    measured_projections(data_path, save_path)
