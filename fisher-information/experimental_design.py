import matplotlib.pyplot as plt
import numpy as np
import os

from typing import List, Callable
from numpy.typing import ArrayLike

from refnx.analysis import Parameter, Objective

from structures import Bilayer
from plotting import save_plot

from simulate import AngleTimes
from simulate import simulate_single_contrast as simulate

from utils import vary_structure
from utils import fisher_single_contrast, fisher_multiple_contrasts

def first_contrast_choice(bilayer: Bilayer, contrasts: ArrayLike,
                          angle_times: AngleTimes, save_path: str) -> None:
    """Investigates how the FI changes, for each parameter of a `bilayer`
       model, for each contrast SLD in a given array of `contrasts`.

    Args:
        bilayer (structures.Bilayer): bilayer to calculate FI with.
        contrasts (numpy.ndarray): contrast SLDs to calculate FI with.
        angle_times (dict): points and simulation times for each angle.
        save_path (str): path to directory to save FI plot to.

    """
    xi = bilayer.parameters

    # Iterate over each SLD in the given array.
    information = []
    for i, contrast_sld in enumerate(contrasts):
        # Simulate data for the given bilayer model with current contrast SLD.
        sample = bilayer.using_contrast(contrast_sld)
        model, data, counts = simulate(sample, angle_times, include_counts=True)

        # Calculate FI matrix and record the diagonal elements.
        g = fisher_single_contrast(data.x, xi, counts, model)
        information.append(np.diag(g))

        # Display progress.
        if i % 50 == 0:
            print('>>> {0}/{1}'.format(i, len(contrasts)))

    # Plot the FI as a function of contrast SLD.
    save_path = os.path.join(save_path, str(bilayer))
    plot_information(contrasts, information, xi, save_path, 'first_contrast', False)
    plot_information(contrasts, information, xi, save_path, 'first_contrast', True)

def second_contrast_choice(bilayer: Bilayer, initial_contrast: float, contrasts: ArrayLike,
                           angle_times: AngleTimes, save_path: str) -> None:
    """Investigates how the FI changes, for each parameter of a `bilayer`
       model, for each second contrast SLD in a given array of `contrasts`.

    Args:
        bilayer (structures.Bilayer): bilayer to find optimal contrast for.
        initial_contrast (float): initial contrast SLD.
        contrasts (numpy.ndarray): second contrast SLDs to calculate FI with.
        angle_times (dict): points and simulation times for each angle.
        save_path (str): path to directory to save FI plot to.

    """
    xi = bilayer.parameters

    # Simulate an experiment using the initial contrast.
    sample = bilayer.using_contrast(initial_contrast)
    model_init, data_init, counts_init = simulate(sample, angle_times, include_counts=True)

    # Iterate over each second contrast SLD in the given array.
    information = []
    for i, contrast_sld in enumerate(contrasts):
        # Simulate data for the bilayer model with current contrast SLD.
        sample = bilayer.using_contrast(contrast_sld)
        model_new, data_new, counts_new = simulate(sample, angle_times, include_counts=True)

        # Combine the data from the initial and second contrasts.
        qs = [data_init.x, data_new.x]
        counts = [counts_init, counts_new]
        models = [model_init, model_new]

        # Calculate the FI matrix for the two contrasts.
        g = fisher_multiple_contrasts(qs, xi, counts, models)
        information.append(np.diag(g))

        # Display progress.
        if i % 50 == 0:
            print('>>> {0}/{1}'.format(i, len(contrasts)))

    # Plot the FI as a function of second contrast SLD.
    save_path = os.path.join(save_path, str(bilayer))
    plot_information(contrasts, information, xi, save_path, 'second_contrast', False)
    plot_information(contrasts, information, xi, save_path, 'second_contrast', True)

def first_angle_choice(structure: Callable, angles: ArrayLike, points: int,
                       time: float, save_path: str) -> None:
    """Investigates how the FI changes, for each parameter of a given
       `structure`, with first angle choice.

    Args:
        structure (function): structure to calculate FI with.
        angles (numpy.ndarray): angles to calculate FI with.
        points (int): points to simulate for each angle.
        time (float): time to use when simulating each angle.
        save_path (str): path to directory to save FI plot to.

    """
    # Iterate over each angle in the given array.
    information = []
    for i, angle in enumerate(angles):
        angle_times = {angle: (points, time)}

        # Simulate data for the given structure with current angle.
        sample = vary_structure(structure())
        model, data, counts = simulate(sample, angle_times, include_counts=True)

        # Get the parameters of the structure.
        objective = Objective(model, data)
        xi = objective.varying_parameters()

        # Calculate FI matrix and individual parameter FIs.
        g = fisher_single_contrast(data.x, xi, counts, model)
        information.append(np.diag(g))

        # Display progress.
        if i % 50 == 0:
            print('>>> {0}/{1}'.format(i, len(angles)))

    # Plot the FI as a function of first angle choice.
    save_path = os.path.join(save_path, structure.__name__)
    plot_information(angles, information, xi, save_path, 'first_angle', False)
    plot_information(angles, information, xi, save_path, 'first_angle', True)

def second_angle_choice(structure: Callable, initial_angle_times: AngleTimes,
                        angles: ArrayLike, points: int, time: float, save_path: str) -> None:
    """Investigates how the FI changes, for each parameter of a given `structure`,
       with second angle choice assuming an `initial_angle` was first measured.

    Args:
        structure (function): structure to calculate the FI with.
        initial_angle_times (dict): angles initially measured.
        angles (numpy.ndarray): angles to calculate FI with.
        points (int): points to simulate for each second angle.
        time (float): time to use when simulating each second angle.
        save_path (str): path to directory to save FI plot to.

    """
    # Simulate the initial angle.
    sample = vary_structure(structure())
    model_init, data_init, counts_init = simulate(sample, initial_angle_times, include_counts=True)

    # Get the parameters of the structure.
    objective = Objective(model_init, data_init)
    xi = objective.varying_parameters()

    # Iterate over each angle in the given array.
    information = []
    for i, angle in enumerate(angles):
        # Simulate an experiment for the new angle using the given points and time.
        new_angle_times = {angle: (points, time)}
        model_new, data_new, counts_new = simulate(sample, new_angle_times, include_counts=True)

        qs = [data_init.x, data_new.x]
        counts = [counts_init, counts_new]
        models = [model_init, model_new]

        # Calculate the FI matrix for the two contrasts.
        g = fisher_multiple_contrasts(qs, xi, counts, models)
        information.append(np.diag(g))

        # Display progress.
        if i % 50 == 0:
            print('>>> {0}/{1}'.format(i, len(angles)))

    # Plot the FI as a function of second angle choice.
    save_path = os.path.join(save_path, structure.__name__)
    plot_information(angles, information, xi, save_path, 'second_angle', False)
    plot_information(angles, information, xi, save_path, 'second_angle', True)

def underlayer_choice(bilayer: Bilayer, thicknesses: ArrayLike, contrast_sld: float,
                      angle_times: AngleTimes, save_path: str) -> None:
    """Investigates how the FI changes, for each parameter of a `bilayer`
       model, with SiO2 underlayer thickness.

    Args:
        bilayer (Bilayer): bilayer to calculate FI with.
        thicknesses (numpy.ndarray): thicknesses to calculate FI over.
        contrast_sld (float): SLD of contrast to simulate.
        angle_times (dict): points and simulation times for each angle.
        save_path (str): path to directory to save FI plot to.

    """
    # Get the underlayer thickness parameter.
    xi = []
    for param in bilayer.parameters:
        if param.name == 'SiO2 Thickness':
            sio2_thick = param
        else:
            xi.append(param)

    # Iterate over each thickness in the given array.
    information = []
    for i, thickness in enumerate(thicknesses):
        # Simulate data for the given bilayer model with current SiO2 thickness.
        sio2_thick.value = thickness
        structure = bilayer.using_contrast(contrast_sld)
        model, data, counts = simulate(structure, angle_times, include_counts=True)

        # Calculate the FI.
        g = fisher_single_contrast(data.x, xi, counts, model)
        information.append(np.diag(g))

        # Display progress.
        if i % 50 == 0:
            print('>>> {0}/{1}'.format(i, len(thicknesses)))

    # Plot the FI as a function of underlayer thickness.
    save_path = os.path.join(save_path, str(bilayer))
    plot_information(thicknesses, information, xi, save_path, 'thickness', False)
    plot_information(thicknesses, information, xi, save_path, 'thickness', True)

def plot_information(x: ArrayLike, information: ArrayLike, xi: List[Parameter],
                     save_path: str, x_label: str, normalise: bool=False) -> None:
    """Plots the FI for each parameter of a model against given `x` array.

    Args:
        x (numpy.ndarray): values to plot FI against.
        information (numpy.ndarray): FI for each parameter.
        xi (list): model parameters.
        save_path (str): path to directory to save FI plot to.
        x_label (str): either 'contrast', 'angle' or 'thickness'.
        normalise (bool): whether to normalise FI to [0,1].

    """
    fig = plt.figure(figsize=[9,7], dpi=600)
    ax = fig.add_subplot(111)

    # Plot the FI for each parameter.
    information = np.asarray(information)
    for i, param in enumerate(xi):
        if normalise: # Normalise to [0,1] using min-max scaling.
            info = information[:,i]
            a = info - np.min(info)
            b = np.max(info) - np.min(info)
            normalised = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
            ax.plot(x, normalised, label=param.name)
        else:
            ax.plot(x, information[:,i], label=param.name)

    ax.legend()

    # Add the correct x-axis label.
    if x_label == 'first_contrast' or x_label == 'second_contrast':
        ax.set_xlabel('$\mathregular{Contrast\ SLD\ (10^{-6} \AA^{-2})}$', fontsize=11, weight='bold')

    elif x_label == 'angle':
        ax.set_xlabel('Angle (°)', fontsize=11, weight='bold')

    elif x_label == 'thickness':
        ax.set_xlabel('$\mathregular{Underlayer\ Thickness\ (\AA)}$', fontsize=11, weight='bold')

    y_label = 'Fisher Information (arb.)'

    if normalise:
        ax.set_ylabel('Normalised '+y_label, fontsize=11, weight='bold')
        save_plot(fig, save_path, '{}_choice_normalised'.format(x_label))
    else:
        # Save both linear and log scale plots.
        ax.set_ylabel(y_label, fontsize=11, weight='bold')
        save_plot(fig, save_path, '{}_choice_linear'.format(x_label))

        ax.set_yscale('log')
        save_plot(fig, save_path, '{}_choice_log'.format(x_label))

def confidence_gain(bilayer: Bilayer, initial_contrast: float, new_contrasts: ArrayLike,
                    angle_times: AngleTimes, save_path: str) -> None:
    """Investigates how the FI confidences ellipses change in size with
       second contrast SLD for a `bilayer`.

    Args:
        bilayer (structures.Bilayer): bilayer to calculate the ellipses with.
        initial_contrast (float): initial contrast SLD.
        new_contrasts (numpy.ndarray): second contrast SLDs.
        angle_times (dict): points and simulation times for each angle.
        save_path (str): path to directory to save plots to.

    """
    xi = bilayer.parameters

    # Simulate an experiment using the initial contrast SLD.
    structure = bilayer.using_contrast(initial_contrast)
    model_init, data_init, counts_init = simulate(structure, angle_times, include_counts=True)

    # Calculate the FI matrix and record the initial confidence ellipse sizes for each parameter pair.
    g = fisher_single_contrast(data_init.x, xi, counts_init, model_init)
    heights_initial = {}
    for i, param in enumerate(xi):
        heights_initial[param] = np.asarray([ellipse_height(g, i, j) for j in range(len(xi))])

    # Iterate over each of the second contrast SLDs.
    heights_new = {param: [] for param in xi}
    for x, new_contrast in enumerate(new_contrasts):
        # Simulate an experiment using the second contrast SLD.
        structure = bilayer.using_contrast(new_contrast)
        model_new, data_new, counts_new = simulate(structure, angle_times, include_counts=True)

        qs = [data_init.x, data_new.x]
        counts = [counts_init, counts_new]
        models = [model_init, model_new]

        # Calculate the FI matrix and new confidence ellipse sizes.
        # Record the difference in size between the initial and new ellipse sizes for each parameter pair.
        g = fisher_multiple_contrasts(qs, xi, counts, models)
        for i, param in enumerate(xi):
            heights = np.asarray([ellipse_height(g, i, j) for j in range(len(xi))])
            heights_new[param].append(heights_initial[param] - heights)

        # Display progress.
        if x % 50 == 0:
            print('>>> {0}/{1}'.format(x, len(new_contrasts)))

    # Plot the reduction in ellipse sizes against contrast SLD.
    save_path = os.path.join(save_path, str(bilayer))
    plot_confidences(new_contrasts, heights_new, save_path)

def ellipse_height(g: ArrayLike, i: int, j: int, k: int=1) -> float:
    """Calculates the size of the semi-minor axis of the confidence ellipse
       between two given parameters with indicies `i` and `j`.

    Args:
        g (numpy.ndarray): FI matrix to calculate ellipses with.
        i (int): index of 1st parameter in matrix.
        j (int): index of 2nd parameter in matrix.
        k (int): size of confidence ellipse in number of standard deviations.

    Returns:
        (float): confidence ellipse semi-minor axis.

    """
    # Get the relevant values from the FI matrix.
    g_params = [[g[i,i], g[i,j]], [g[j,i], g[j,j]]]

    # Initialise min and max variables.
    x_min, x_max = float('inf'), -float('inf')
    min_coords, max_coords = None, None

    # Iterate over each coordinate of the confidence ellipse.
    for theta in np.arange(0, 2*np.pi, 0.001):
        X = np.asarray([np.sin(theta), np.cos(theta)])
        epsilon = k / np.sqrt(np.dot(np.dot(X, g_params), X.T))
        x = epsilon*np.sin(theta)

        # Record the points with minimum and maximum x.
        if x <= x_min:
            y = epsilon*np.cos(theta)
            min_coords = np.array((x,y))
            x_min = x
        if x >= x_max:
            x = epsilon*np.cos(theta)
            max_coords = np.array((x,y))
            x_max = x

    # Return the distance between the min and max points.
    return np.linalg.norm(max_coords-min_coords)

def plot_confidences(contrasts: ArrayLike, confidence_gains: ArrayLike, save_path: str) -> None:
    """Plots the reduction in confidence ellipse size as a function of second
       contrast SLD for each parameter pair.

    Args:
        contrasts (numpy.ndarray): second contrast SLDs.
        confidence_gains (numpy.ndarray): reduction in confidence ellipse sizes.
        save_path (string): path to directory to save plots to.

    """
    # Iterate over each parameter.
    labels = [param.name for param in confidence_gains]
    for i, param in enumerate(confidence_gains):
        fig = plt.figure(figsize=[9,7], dpi=600)
        ax = fig.add_subplot(111)

        # Iterate over all other parameters.
        gain = np.array(confidence_gains[param])
        for j in range(gain.shape[1]):
            # Plot the confidence gain as a function of contrast SLD for the parameter pair.
            if i != j: # Do not plot a parameter against itself.
                ax.plot(contrasts, gain[:,j], label=labels[j])

        # Set the plot title to the first parameter in each parameter pair.
        ax.set_title(param.name)
        ax.legend(loc='upper right')
        ax.set_xlabel("$\mathregular{Contrast\ SLD\ (10^{-6} \AA^{-2})}$", fontsize=11, weight='bold')
        ax.set_ylabel("Confidence Gain", fontsize=11, weight='bold')

        save_plot(fig, save_path, 'confidence_gain_{}'.format(i+1))

if __name__ == '__main__':
    from structures import SymmetricBilayer
    from structures import SingleAsymmetricBilayer, DoubleAsymmetricBilayer

    save_path = './results'

    bilayer = SymmetricBilayer()
    initial_contrast = 6.36
    contrasts = np.arange(-0.56, 6.35, 0.01)
    angle_times = {0.7: (70, 10),
                   2.0: (70, 40)}

    # Investigate how the FI changes with initial contrast SLD.
    first_contrast_choice(bilayer, contrasts, angle_times, save_path)

    # Investigate how the FI changes with second contrast SLD.
    second_contrast_choice(bilayer, initial_contrast, contrasts, angle_times, save_path)

    # Investigate how the FI changes with SiO2 underlayer thickness.
    thicknesses = np.arange(1, 500, 0.5)
    underlayer_choice(bilayer, thicknesses, initial_contrast, angle_times, save_path)

    from structures import similar_sld_sample_1, similar_sld_sample_2
    from structures import thin_layer_sample_1, thin_layer_sample_2
    from structures import easy_sample, QCS_sample, many_param_sample

    structure = easy_sample
    points = 70
    time = 1
    angles = np.arange(0.2, 2.4, 0.01)

    # Investigate how the FI changes with first angle choice.
    first_angle_choice(structure, angles, points, time, save_path)

    # Investigate how the FI changes with second angle choice.
    angle_times = {0.7: (70, 1)} # First angle choice.
    second_angle_choice(structure, angle_times, angles, points, time, save_path)
