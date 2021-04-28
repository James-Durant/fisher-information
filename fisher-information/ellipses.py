import matplotlib.pyplot as plt
import numpy as np
import os, sys
sys.path.append("./") # MCMC sampling cannot find structures code without this?

from typing import List, Callable
from numpy.typing import ArrayLike

from refnx.analysis import Parameter, Objective

from plotting import plot_objective, save_plot

from utils import vary_structure, Sampler
from utils import fisher_single_contrast as fisher

from simulate import AngleTimes
from simulate import simulate_single_contrast as simulate

def compare_ellipses(structure: Callable, angle_times: AngleTimes, save_path: str) -> None:
    """Calculates parameter uncertainties for a given `structure` using MCMC
       and nested sampling, calculates the FIM and plots the FIM confidence
       ellipses on the sampling corner plots.

    Args:
        structure (function): structure to simulate with.
        angle_times (dict): points and simulation times for each angle.
        save_path (str): path to directory to save data and figures.

    """
    save_path = os.path.join(save_path, structure.__name__)

    # Simulate an experiment using the given angles, number of points and times.
    structure = vary_structure(structure())
    model, data, counts = simulate(structure, angle_times, include_counts=True, save_path=save_path)

    # Get the parameters of the structure.
    objective = Objective(model, data)
    xi = objective.varying_parameters()

    sampler = Sampler(objective)

    # Run MCMC sampling and get the resulting corner plot.
    fig1 = sampler.sample_MCMC(verbose=True)
    fig2, _ = plot_objective(objective)

    # Calculate the FIM matrix and plot confidence ellipses on corner plots.
    g = fisher(data.x, xi, counts, model)
    plot_ellipses(g, xi, fig1)
    save_plot(fig1, save_path, 'confidence_ellipses_MCMC')
    save_plot(fig2, save_path, 'fit_MCMC')

    # Run nested sampling and overlay confidence ellipses on corner plot.
    fig3 = sampler.sample_nested(verbose=True, dynamic=True)
    fig4, _ = plot_objective(objective)

    g = fisher(data.x, xi, counts, model)
    plot_ellipses(g, xi, fig3)
    save_plot(fig3, save_path, 'confidence_ellipses_nested')
    save_plot(fig4, save_path, 'fit_nested')

def plot_ellipses(g: ArrayLike, xi: List[Parameter], fig: plt.Figure) -> None:
    """Overlays the FIM confidence ellipses on the corner plot from either
       MCMC or nested sampling.

    Args:
        g (numpy.ndarray): Fisher information metric matrix.
        xi (list): varying parameters.
        fig (matplotlib.pyplot.figure): sampling corner plot.

    """
    # Get the existing axes from the corner plot.
    axes = np.reshape(np.array(fig.get_axes()), (len(xi), len(xi)))
    m = len(xi)
    for i in range(m):
        for j in range(m):
            if i > j: # Plot the confidence ellipse on plots below diagonal.
                confidence_ellipse(g, j, i, xi[j], xi[i],
                                   axes[i,j], i==m-1, j==0)
            elif i == j:
                continue # Leave diagonal plots as they are.
            else:
                axes[i,j].set_visible(False) # Remove all other plots.

    axes[m-1, m-1].set_xlabel(xi[-1].name)

def confidence_ellipse(g: ArrayLike, i: int, j: int, param1: Parameter, param2: Parameter,
                       axis: plt.Axes, show_xlabel: bool, show_ylabel: bool) -> None:
    """Plots the confidence ellipse between `param1` and `param2`.

    Args:
        g (numpy.ndarray): Fisher information metric matrix.
        i (int): index of `param1` in the FIM.
        j (int): index of `param2` in the FIM.
        param1 (refnx.analysis.Parameter): 1st parameter corresponding to `i`.
        param2 (refnx.analysis.Parameter): 2nd parameter corresponding to `j`.
        axis (matplotlib.pyplot.Axes): subplot of corner plot to plot on.
        show_xlabel (bool): whether to display x-axis label.
        show_ylabel (bool): whether to display y-axis label.

    """
    # Retrieve the elements of the FIM for the given parameters.
    g_params = [[g[i,i], g[i,j]], [g[j,i], g[j,j]]]

    # Iterative over k: the number of standard deviations.
    for k in [1,2,3]:
        # Calculate the values of the confidence ellipse.
        x, y = [], []
        for theta in np.arange(0, 2*np.pi, 0.001):
            X = np.array([np.sin(theta), np.cos(theta)])
            epsilon = k / np.sqrt(np.dot(np.dot(X, g_params), X.T))
            x.append(epsilon*np.sin(theta))
            y.append(epsilon*np.cos(theta))

        # Move the confidence ellipse to be centred on the parameter estimates.
        x = np.array(x) + param1.value
        y = np.array(y) + param2.value
        axis.plot(x,y, color='r')

    if show_xlabel:
        axis.set_xlabel(param1.name)
    if show_ylabel:
        axis.set_ylabel(param2.name)

if __name__ == '__main__':
    from structures import similar_sld_sample_1, similar_sld_sample_2
    from structures import thin_layer_sample_1, thin_layer_sample_2
    from structures import easy_sample, QCS_sample, many_param_sample

    save_path = './results'

    structure   = easy_sample # Choose structure here.
    angle_times = {0.7: (70, 5), # Angle: (Points, Time)
                   2.0: (70, 20)}

    compare_ellipses(structure, angle_times, save_path)
