import matplotlib.pyplot as plt
import numpy as np
import os

from typing import List, Callable
from numpy.typing import ArrayLike

from refnx.analysis import Parameter, Objective, GlobalObjective, CurveFitter

from structures import Bilayer
from simulate import simulate_single_contrast, AngleTimes

from utils import vary_structure, get_ground_truths, Sampler
from plotting import save_plot

def fitting_biases(structure: Callable, angle_times: AngleTimes,
                   save_path: str, n: int=500) -> None:
    """Investigates fitting biases in nested sampling and differential
       evolution with and without following up with L-BFGS-B.

    Args:
        structure (function): structure to investigate biases with.
        angle_times (dict): points and measurement times for each angle.
        save_path (str): path to directory to save bias results to.
        n (int): number of fits to calculate bias with.

    """
    true = get_ground_truths(structure)

    evolution_params, lbfgs_params, sampled_params = [], [], []
    for i in range(n): # Fit n times.
        # Simulate an experiment, vary the model parameters.
        objective = Objective(*simulate_single_contrast(structure(),
                                                        angle_times))
        # Randomly initialise.
        vary_structure(objective.model.structure, random_init=True)

        xi = objective.varying_parameters()

        # Fit only first and record the estimated values.
        fitter = CurveFitter(objective)
        fitter.fit('differential_evolution', verbose=False, polish=False)
        evolution_params.append([param.value for param in xi])

        # Then fit using L-BFGS-B and record the estimated parameter values.
        fitter.fit('L-BFGS-B', verbose=False)
        lbfgs_params.append([param.value for param in xi])

        # Sample using nested sampling and record the estimated values.
        Sampler(objective).sample_nested(verbose=False, show_fig=False)
        sampled_params.append([param.value for param in xi])

        # Display progress.
        print('>>> {0}/{1}'.format(i+1, n))

    # Calculate the biases over the `n` fits.
    evolution_biases = np.mean(evolution_params, axis=0) - true
    lbfgs_biases = np.mean(lbfgs_params, axis=0) - true
    sampling_biases = np.mean(sampled_params, axis=0) - true

    names = [param.name for param in objective.varying_parameters()]

    # Save the biases for each parameter for the three fitting methods.
    save_path = os.path.join(save_path, structure.__name__,)
    save_biases(evolution_biases, names,
                'Differential Evolution (No Polish)', save_path)
    save_biases(lbfgs_biases, names,
                'Differential Evolution + L-BFGS-B', save_path)
    save_biases(sampling_biases, names, 'Sampling', save_path)

def save_biases(biases: ArrayLike, names: List[str], method: str,
                save_path: str) -> None:
    """Saves calculated biases to a .txt file.

    Args:
        biases (numpy.ndarray): array of biases to save.
        names (list): list of parameter names.
        method (str): fitting method used to calculate the biases.
        save_path (str): path to directory to save the bias results to.

    """
    # Create the directory if not present.
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Save the biases to a .txt file.
    # Create a new file if not present. Otherwise, append to it.
    file_path = os.path.join(save_path, 'fitting_biases.txt')
    with open(file_path, 'a' if os.path.exists(file_path) else 'w') as file:
        file.write(method+'\n')
        for name, bias in list(zip(names, biases)):
            file.write('{0}: {1}\n'.format(name, bias))
        file.write('\n')

def time_biases(structure: Callable, angle_times: AngleTimes,
                multipliers: ArrayLike, save_path: str, n: int=100) -> None:
    """Investigates how fitting biases change with measurement time.

    Args:
        structure (function): structure to investigate the biases with.
        angle_times (dict): points and measurement times for each angle.
        multipliers (numpy.ndarray): array of time multipliers.
        save_path (str): path to directory to save the bias plot to.
        n (int): number of fits to calculate the bias with.

    """
    # Get the true values of the layers' thicknesses and SLDs.
    true = get_ground_truths(structure)

    # Iterate over each time (multiplier).
    biases = []
    for i, multiplier in enumerate(multipliers):
        fitted_params = []

        # Fit n times using the current time multiplier.
        for _ in range(n):
            # Multiply the initial times by current multiplier for each angle.
            new_angle_times = {angle: (angle_times[angle][0],
                                       angle_times[angle][1]*multiplier)
                               for angle in angle_times}

            # Simulate an experiment, vary the model parameters.
            objective = Objective(*simulate_single_contrast(structure(),
                                                            new_angle_times))
            # Randomly initialise.
            vary_structure(objective.model.structure, random_init=True)

            # Fit the objective and record the estimated parameter values.
            xi = objective.varying_parameters()
            CurveFitter(objective).fit('differential_evolution', verbose=False)
            fitted_params.append([param.value for param in xi])

        # Calculate the bias over the n fits.
        biases.append(np.mean(fitted_params, axis=0) - true)

        # Display progress.
        print('>>> {0}/{1}'.format(i+1, len(multipliers)))

    # Plot the fitting biases as a function of measurement time.
    save_path = os.path.join(save_path, structure.__name__)
    plot_biases(multipliers, biases, xi, 'Time Multiplier',
                save_path, 'time_biases')

def contrast_biases(bilayer: Bilayer, initial_contrast: float,
                    new_contrasts: ArrayLike, angle_times: AngleTimes,
                    save_path: str, n: int=10) -> None:
    """Investigates how fitting biases change with second contrast choice
       for a bilayer model.

    Args:
        bilayer (structures.Bilayer): bilayer to calculate biases with.
        initial_contrast (float): initial measurement contrast.
        new_contrasts (numpy.ndarray): second contrasts to measure.
        angle_times (dict): points and measurement times for each angle.
        save_path (str): path to directory to save bias plot to.
        n (int): number of fits to calculate bias with.

    """
    xi = bilayer.parameters

    # Simulate an experiment using the initial measurement contrast.
    structure = bilayer.using_contrast(initial_contrast)
    objective_initial = Objective(*simulate_single_contrast(structure,
                                                            angle_times))
    # Ground truth parameter values.
    true = np.asarray([param.value for param in xi])

    # Iterate over each second measurement contrast.
    biases = []
    for x, new_contrast in enumerate(new_contrasts, 1):
        fitted = []

        # Calculate the bias over n fits.
        for _ in range(n):
            # Set the bilayer parameters back to their ground truth values.
            for i, param in enumerate(xi):
                param.value = true[i]

            # Simulate an experiment using the second measurement contrast.
            structure = bilayer.using_contrast(new_contrast)
            objective_new = Objective(*simulate_single_contrast(structure,
                                                                angle_times))

            # Randomly initialise the parameter values.
            for param in xi:
                param.value = np.random.uniform(param.bounds.lb,
                                                param.bounds.ub)

            # Fit the objective and record the estimated parameter values.
            objectives = [objective_initial, objective_new]
            fitter = CurveFitter(GlobalObjective(objectives))
            fitter.fit('differential_evolution', verbose=False)
            fitted.append([param.value for param in xi])

        # Calculate the bias over the n fits.
        biases.append(np.mean(fitted, axis=0) - true)

        # Display progress.
        print('>>> {0}/{1}'.format(x, len(new_contrasts)))

    # Plot the fitting biases as a function of second contrast choice.
    save_path = os.path.join(save_path, str(bilayer))
    plot_biases(new_contrasts, biases, xi,
                '$\mathregular{Contrast\ SLD\ (10^{-6} \AA^{-2})}$',
                save_path, 'contrast_biases')

def plot_biases(x: ArrayLike, biases: ArrayLike, xi: List[Parameter],
                x_label: str, save_path: str, file_name: str) -> None:
    """Plots biases against either measurement time or contrast choice.

    Args:
        x (numpy.ndarray): either measurement times or contrast SLDs.
        biases (numpy.ndarray): biases corresponding to x array.
        x_label (str): label to use for x-axis
        save_path (str): path to directory to save bias plot to.
        file_name (str): name to use for bias plot file.

    """
    fig = plt.figure(figsize=[9,7], dpi=600)
    ax = fig.add_subplot(111)

    # Plot the biases for each parameter.
    biases = np.asarray(biases)
    for i, param in enumerate(xi):
        ax.plot(x, biases[:,i], label=param.name)

    ax.set_xlabel(x_label, fontsize=11, weight='bold')
    ax.set_ylabel('Bias', fontsize=11, weight='bold')
    ax.legend(loc='upper right')

    save_plot(fig, save_path, file_name) # Save the plot.

def bias_derivative(bilayer: Bilayer, initial_contrast: float,
                    new_contrasts: ArrayLike, param_name: str,
                    param_range: ArrayLike, angle_times: AngleTimes,
                    save_path: str, n: int=50):
    """Investigates how fitting biases change with second contrast choice in
       a chosen parameter for a bilayer model. This is repeated with the
       parameter of choice set to each value in the given `param_range`.

    Args:
        bilayer (structures.Bilayer): bilayer to calculate biases with.
        initial_contrast (float): initial measurement contrast.
        new_contrasts (numpy.ndarray): second contrasts to measure.
        param_name (str): name of parameter to vary.
        param_range (numpy.ndarray): values to set the varying parameter to.
        angle_times (dict): points and measurement times for each angle.
        save_path (str): path to directory to save bias plot to.
        n (int): number of fits to calculate bias with.

    """
    # Get the parameter of interest from the model.
    parameter = None
    for param in bilayer.parameters:
        if param.name == param_name:
            parameter = param

    if parameter is None:
        # `param_name` does not match any parameters in the `bilayer` model.
        raise RuntimeError('parameter not in model')

    biases = []
    for value in param_range:
        # Set the chosen parameter to each value in the given range.
        parameter.value = value
        # Set bound on the parameter to 25 % above and below the true value.
        parameter.setp(vary=True, bounds=(value*0.75, value*1.25))

        # Simulate an experiment using the initial contrast with this value.
        structure = bilayer.using_contrast(initial_contrast)
        model, data = simulate_single_contrast(structure, angle_times)
        objective_initial = Objective(model, data)

        # Iterate over each second contrast SLD to measure.
        value_biases = []
        for i, new_contrast in enumerate(new_contrasts, 1):
            fitted = []

            # Calculate fitting bias over `n` fits with this parameter value.
            for _ in range(n):
                # Simulate experiment using the second measurement contrast.
                structure = bilayer.using_contrast(new_contrast)
                model, data = simulate_single_contrast(structure, angle_times)
                objective_new = Objective(model, data)

                # Set the parameter to a range value within its bounds.
                parameter.value = np.random.uniform(parameter.bounds.lb,
                                                    parameter.bounds.ub)

                # Fit the initial and new contrast data.
                objectives = [objective_initial, objective_new]
                global_objective = GlobalObjective(objectives)
                global_objective.varying_parameters = lambda: [parameter]

                fitter = CurveFitter(global_objective)
                fitter.fit('differential_evolution', verbose=False)

                # Record the fitted value.
                fitted.append(parameter.value)

            # Calculate the bias from the `n` fits.
            value_biases.append(np.mean(fitted) - value)

            # Display progress.
            print(">>> {0}/{1}".format(i, len(new_contrasts)))

        # Record the biases for each contrast for this parameter value.
        biases.append(value_biases)
        print()

    # Create plot.
    fig = plt.figure(figsize=[9,7], dpi=600)
    ax = fig.add_subplot(111)

    # Plot bias vs. second measurement contrast SLD for each parameter value.
    for i, value in enumerate(param_range):
        ax.plot(new_contrasts, biases[i], label=param_name+'='+str(value))

    ax.set_xlabel("$\mathregular{Contrast\ SLD\ (10^{-6} \AA^{-2})}$",
                  fontsize=11, weight='bold')
    ax.set_ylabel("$\mathregular{Bias\ (\AA)}$", fontsize=11, weight='bold')
    ax.legend(loc='upper right')

    # Save the plot.
    save_path = os.path.join(save_path, str(bilayer))
    file_name = param_name.lower().replace(' ', '-')
    save_plot(fig, save_path, "{}_bias_derivative".format(file_name))

if __name__ == '__main__':
    from structures import similar_sld_sample_1, similar_sld_sample_2
    from structures import thin_layer_sample_1, thin_layer_sample_2
    from structures import easy_sample, QCS_sample, many_param_sample

    save_path = './results'

    structure = easy_sample
    angle_times = {0.7: (70, 5),
                   2.0: (70, 20)}

    # Investigate biases in fitting and sampling.
    fitting_biases(structure, angle_times, save_path, 1000)

    # Investigate how biases change with measurement time.
    multipliers = np.arange(1, 10, 0.2)
    time_biases(structure, angle_times, multipliers, save_path, 50)

    from structures import SymmetricBilayer
    from structures import SingleAsymmetricBilayer, DoubleAsymmetricBilayer

    bilayer = SingleAsymmetricBilayer()
    initial = 6.36
    contrasts = np.arange(-0.56, 6.36, 0.25)
    angle_times = {0.7: (70, 10),
                   2.0: (70, 40)}

    # Investigate how biases change with measurement contrast.
    contrast_biases(bilayer, initial, contrasts, angle_times, save_path, 50)

    # Investigate the bias in the chosen parameter for each value of the
    # parameter in the given parameter range.
    param_name = 'Inner Headgroup Thickness'
    param_range = np.arange(12, 22, 1)
    bias_derivative(bilayer, initial, contrasts, param_name, param_range,
                    angle_times, save_path, 50)
