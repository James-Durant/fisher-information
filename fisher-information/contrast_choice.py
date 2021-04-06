import matplotlib.pyplot as plt
import numpy as np
import os

from typing import List, Tuple, Dict
from numpy.typing import ArrayLike

from refnx.analysis import Parameter

from structures import Bilayer
from simulate import simulate_single_contrast, AngleTimes

from plotting import save_plot
from utils import fisher_single_contrast, fisher_multiple_contrasts

def contrast_choice(bilayer: Bilayer, contrasts: ArrayLike,
                    angle_times: AngleTimes, save_path: str) -> None:
    """Investigates how the FIM changes, for each parameter of a bilayer model,
       for each contrast SLD in a given array of contrasts.

    Args:
        bilayer (Bilayer): bilayer model to find optimal contrast for.
        contrasts (numpy.ndarray): contrast SLDs to calculate FIM with.
        angle_times (dict): points and measurement times for each angle.
        save_path (str): path to directory to save FIM plot to.

    """
    save_path = os.path.join(save_path, str(bilayer))
    xi = bilayer.parameters

    # Iterate over each SLD in the given array.
    information = []
    for i, contrast_sld in enumerate(contrasts):
        # Simulate data for the given bilayer model with current contrast SLD.
        structure = bilayer.using_contrast(contrast_sld)
        model, data, counts = simulate_single_contrast(structure, angle_times,
                                                       include_counts=True)
        # Calculate the FIM
        g = fisher_single_contrast(data.x, xi, counts, model)
        information.append(np.diag(g))

        # Display progress.
        print('>>> {0}/{1}'.format(i+1, len(contrasts)))

    # Plot the FIM as a function of contrast SLD.
    plot_information(information, xi, save_path, normalise=False)
    plot_information(information, xi, save_path, normalise=True)

def plot_information(information: ArrayLike, xi: List[Parameter],
                     save_path: str, normalise: bool=False) -> None:
    """Plots the FIM for each parameter of a bilayer model against contrast SLD.

    Args:
        information (numpy.ndarray): FIM values for each parameter.
        xi (list): model parameters.
        save_path (str): path to directory to save FIM plot to.
        normalise (bool): whether to normalise the information values to [0,1].

    """
    fig = plt.figure(figsize=[9,7], dpi=600)
    ax = fig.add_subplot(111)

    # Plot the FIM values for each free parameter.
    information = np.asarray(information)
    for i, param in enumerate(xi):
        if normalise: # Normalise to [0,1] using min-max scaling.
            info = information[:,i]
            normalised = (info - np.min(info)) / (np.max(info) - np.min(info))
            ax.plot(contrasts, normalised, label=param.name)
        else:
            ax.plot(contrasts, information[:,i], label=param.name)

    ax.set_xlabel('$\mathregular{Contrast\ SLD\ (10^{-6} \AA^{-2})}$',
                  fontsize=11, weight='bold')
    ax.set_ylabel('Fisher Information', fontsize=11, weight='bold')
    ax.legend()

    if normalise:
        ax.set_ylabel('Normalised Fisher Information',
                      fontsize=11, weight='bold')
        save_plot(fig, save_path, 'contrast_choice_normalised')
    else:
        # Save both a linear and log scale plot.
        ax.set_ylabel('Fisher Information', fontsize=11, weight='bold')
        save_plot(fig, save_path, 'contrast_choice_linear')

        ax.set_yscale('log')
        save_plot(fig, save_path, 'contrast_choice_log')

def confidence_gain(bilayer: Bilayer, initial_contrast: float,
                    new_contrasts: ArrayLike, angle_times: AngleTimes,
                    save_path: str) -> None:
    """Investigates how the FIM confidences ellipses change in size with second
       measurement contrast SLD.

    Args:
        bilayer (Bilayer): bilayer model to calculate ellipses on.
        initial_contrast (float): initial measured contrast for the bilayer.
        new_contrasts (numpy.ndarray): second measurement contrasts.
        angle_times (dict): points and measurement times for each angle.
        save_path (str): path to directory to save plots to.

    """
    save_path = os.path.join(save_path, str(bilayer))
    xi = bilayer.parameters

    # Simulate an experiment using the initial measured contrast.
    structure = bilayer.using_contrast(initial_contrast)
    simulated_initial = simulate_single_contrast(structure, angle_times,
                                                 include_counts=True)
    model_initial, data_initial, counts_initial = simulated_initial

    # Calculate the Fisher information matrix and record the initial
    # confidence ellipse sizes for each parameter pair.
    g = fisher_single_contrast(data_initial.x, xi, counts_initial, model_initial)
    heights_initial = {}
    for i, param in enumerate(xi):
        heights_initial[param] = np.asarray([ellipse_height(g, i, j)
                                             for j in range(len(xi))])

    # Iterate over each of the second contrasts to measure.
    heights_new = {param: [] for param in xi}
    for x, new_contrast in enumerate(new_contrasts):
        # Simulate an experiment using the second contrast.
        structure = bilayer.using_contrast(new_contrast)
        simulated_new = simulate_single_contrast(structure, angle_times,
                                                 include_counts=True)
        model_new, data_new, counts_new = simulated_new

        qs = [data_initial.x, data_new.x]
        counts = [counts_initial, counts_new]
        models = [model_initial, model_new]

        # Calculate the Fisher information matrix calculate and calculate the
        # new confidence ellipse sizes.
        # Record the difference in size between the initial and new ellipse
        # sizes for each parameter pair.
        g = fisher_multiple_contrasts(qs, xi, counts, models)
        for i, param in enumerate(xi):
            heights = np.asarray([ellipse_height(g, i, j) for j in range(len(xi))])
            heights_new[param].append(heights_initial[param] - heights)

        # Display progress.
        print('>>> {0}/{1}'.format(x+1, len(new_contrasts)))

    # Plot the reduction in ellipse sizes against contrast SLD.
    plot_confidences(new_contrasts, heights_new, save_path)

def ellipse_height(g: ArrayLike, i: int, j: int, k: int=1) -> float:
    """Calculates the size of the semi-minor axis of the confidence ellipse
       between two given parameters.

    Args:
        g (numpy.ndarray): Fisher information matrix to calculate ellipses with.
        i (int): index of 1st parameter in the matrix.
        j (int): index of 2nd parameter in the matrix.
        k (int): size of confidence ellipse in number of standard deviations.

    Returns:
        float: confidence ellipse semi-minor axis

    """
    # Get the relevant values from the Fisher information matrix.
    g_params = [[g[i,i], g[i,j]], [g[j,i], g[j,j]]]

    # Initialise min and max variables.
    x_min, x_max = float('inf'), -float('inf')
    min_coords, max_coords = None, None

    # Iterate over each coordinate of the confidence ellipse.
    for theta in np.arange(0, 2*np.pi, 0.001):
        X = np.asarray([np.sin(theta), np.cos(theta)])
        epsilon = k / np.sqrt(np.dot(np.dot(X, g_params), X.T))
        x = epsilon*np.sin(theta)

        # Record the points with maximum and minimum x.
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

def plot_confidences(contrasts: ArrayLike, confidence_gains: ArrayLike,
                     save_path: str) -> None:
    """Plots the reduction in confidence ellipse size as a function of second
       measurement contrast SLD for each parameter pair.

    Args:
        contrasts (numpy.ndarray): second measurement contrasts.
        confidence_gains (numpy.ndarray): reductions in ellipse size.
        save_path (string): path to directory to save plots to.

    """
    # Iterate over each parameter.
    labels = [param.name for param in confidence_gains]
    for i, param in enumerate(confidence_gains):
        fig = plt.figure(figsize=[9,7], dpi=600)
        ax  = fig.add_subplot(111)

        # Iterate over all other parameters.
        gain = np.array(confidence_gains[param])
        for j in range(gain.shape[1]):
            # Plot the confidence gain as a function of contrast SLD
            # for the parameter pair.
            if i != j:
                ax.plot(contrasts, gain[:,j], label=labels[j])

        # Set the plot title to the first parameter in each parameter pair.
        ax.set_title(param.name)

        ax.set_xlabel("$\mathregular{Contrast\ SLD\ (10^{-6} \AA^{-2})}$",
                      fontsize=11, weight='bold')
        ax.set_ylabel("Confidence Gain", fontsize=11, weight='bold')
        ax.set_yscale('log')
        ax.legend()
        save_plot(fig, save_path, 'confidence_gain_{}'.format(i))

if __name__ == '__main__':
    from structures import SymmetricBilayer
    from structures import SingleAsymmetricBilayer, DoubleAsymmetricBilayer

    save_path = './results'

    bilayer = SymmetricBilayer()
    angle_times = {0.7: (70, 10),
                   2.0: (70, 40)}

    # Investigate how Fisher information changes with initial contrast SLD.
    contrasts = np.arange(-0.56, 6.35, 0.01)
    contrast_choice(bilayer, contrasts, angle_times, save_path)

    # Investigate how ellipse sizes change with second measured contrast SLD.
    initial = 6.36
    confidence_gain(bilayer, initial, contrasts, angle_times, save_path)
