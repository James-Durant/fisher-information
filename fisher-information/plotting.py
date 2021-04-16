import matplotlib.pyplot as plt
import numpy as np
import os

from typing import List, Tuple
from numpy.typing import ArrayLike

from refnx.dataset import ReflectDataset
from refnx.reflect import Structure, ReflectModel
from refnx.analysis import Objective

def plot_sld_profile(structure: Structure, distances: ArrayLike=None,
                     colour: str='black', label: bool=False
                     ) -> Tuple[plt.Figure, plt.Axes]:
    """Plots the SLD profile of a given structure.

    Args:
        structure (refnx.reflect.Structure): structure to plot SLD profile of.
        distances (numpy.ndarray): range of values to use on the x-axis.
        colour (str): colour to use for the SLD profile.
        label (bool): whether to use structure's name in plot's legend.

    Returns:
        fig (matplotlib.pyplot.Figure): figure containing plotted SLD profile.
        ax (matplotlib.pyplot.Axes): axis containing plotted SLD profile.

    """
    fig = plt.figure(figsize=[9,7], dpi=600)
    ax = fig.add_subplot(111)

    # Plot the SLD profile with or without a label.
    ax.plot(*structure.sld_profile(distances), color=colour,
            label=structure.name if label else None)

    ax.set_xlabel('$\mathregular{Distance\ (\AA)}$',
                  fontsize=11, weight='bold')
    ax.set_ylabel('$\mathregular{SLD\ (10^{-6} \AA^{-2})}$',
                  fontsize=11, weight='bold')
    return fig, ax

def plot_sld_profiles(structures: List[Structure], distances: ArrayLike=None,
                      label: bool=True) -> Tuple[plt.Figure, plt.Axes]:
    """Plots the SLD profiles of a given list structures.

    Args:
        structures (list): structures to plot SLD profiles of.
        distances (numpy.ndarray): range of values to use on the x-axis.
        label (bool): whether to use the structures' name in the plot's legend.

    Returns:
        fig (matplotlib.pyplot.Figure): figure containing plotted SLD profiles.
        ax (matplotlib.pyplot.Axes): axis containing plotted SLD profiles.

    """
    # Get the figure and axis by plotting the first structure by itself.
    fig, ax = plot_sld_profile(structures[0], distances, None, label)

    # Plot the rest of the structures on the same axis.
    for structure in structures[1:]:
        ax.plot(*structure.sld_profile(distances),
                label=structure.name if label else None)

    if label: # If labelling, create the legend.
        ax.legend()

    return fig, ax

def plot_objective(objective: Objective, colour: str='black',
                   label: bool=False) -> Tuple[plt.Figure, plt.Axes]:
    """Plots the fit of a given objective against the objective's data.

    Args:
        objective (refnx.analysis.Objective): objective to plot.
        colour (str): colour to use for objective's data points.
        label (bool): whether to use structure's name in the plot's legend.

    Returns:
        fig (matplotlib.pyplot.Figure): figure containing plotted objective.
        ax (matplotlib.pyplot.Axes): axis containing plotted objective.

    """
    # Plot the reflectivity data.
    fig, ax = plot_refdata(objective.data, colour,
                           objective.model.structure.name if label else None)

    # Plot the fit.
    q = objective.data.x
    ax.plot(q, objective.model(q), color='red', zorder=20)

    return fig, ax

def plot_objectives(objectives: List[Objective], label: bool=True
                    ) -> Tuple[plt.Figure, plt.Axes]:
    """Plots fits of a given list of objectives against the objectives' data.

    Args:
        objectives (list): list of objectives to plot.
        label (bool): whether to include a legend in the plot.

    Returns:
        fig (matplotlib.pyplot.Figure): figure containing plotted objectives.
        ax (matplotlib.pyplot.Axes): axis containing plotted objectives.

    """
    # Get the figure and axis by plotting the first objective by itself.
    fig, ax = plot_objective(objectives[0], None, label)

    # Plot the rest of the objectives on the same axis.
    for objective in objectives[1:]:
        q, r = objective.data.x, objective.data.y
        r_error = objective.data.y_err

        # Plot the reflectivity data.
        ax.errorbar(q, r, r_error,
                    marker='o', ms=3, lw=0, elinewidth=1, capsize=1.5,
                    label=objective.model.structure.name if label else None)

        # Plot the fit.
        ax.plot(q, objective.model(q), color='red', zorder=20)

    if label: # If labelling, create the legend.
        ax.legend()

    return fig, ax

def plot_refdata(data: ReflectDataset, colour: str='black',
                 label: bool=None) -> Tuple[plt.Figure, plt.Axes]:
    """Plots a given reflectivity dataset.

    Args:
        data (refnx.dataset.ReflectDataset): dataset to plot.
        colour (str): colour to use when plotting the data points.
        label (str): label of dataset for inclusion in plot's legend.

    Returns:
        fig (matplotlib.pyplot.Figure): figure containing reflectivity data.
        ax (matplotlib.pyplot.Axes): axis containing reflectivity data.

    """
    fig = plt.figure(figsize=[9,7], dpi=600)
    ax = fig.add_subplot(111)

    # Plot the reflectivity data (Q, R, dR).
    ax.errorbar(data.x, data.y, data.y_err, color=colour, marker='o',
                ms=3, lw=0, elinewidth=1, capsize=1.5, label=label)

    ax.set_xlabel('$\mathregular{Q\ (Å^{-1})}$', fontsize=11, weight='bold')
    ax.set_ylabel('Reflectivity (arb.)', fontsize=11, weight='bold')
    ax.set_yscale('log')
    #ax.set_xlim(0, 0.3)
    ax.set_ylim(1e-7, 2)
    return fig, ax

def plot_reflectivity_curve(structure: Structure, q_min: float=0.005,
                            q_max: float=0.3, points: int=500, dq: float=2,
                            bkg: float=1e-7) -> Tuple[plt.Figure, plt.Axes]:
    """Plots the model reflectivity curve of a given structure.

    Args:
        structure (refnx.reflect.Structure): structure to plot reflectivity
                                             curve of.
        q_min (float): minimum Q value for the plot.
        q_max (float): maximum Q value for the plot.
        points (int): number of reflectivity points to plot.
        dq (float): instrument resolution parameter.
        bkg (float): instrument background parameter.

    Returns:
        fig (matplotlib.pyplot.Figure): figure containing reflectivity curve.
        ax (matplotlib.pyplot.Axes): axis containing reflectivity curve.

    """
    model = ReflectModel(structure, scale=1, bkg=bkg, dq=dq) # Define a model.
    q = np.logspace(np.log10(q_min), np.log10(q_max), points)
    r = model(q) # Calculate the model reflectivity.

    # Plot the model reflectivity against Q.
    fig = plt.figure(figsize=[9,7], dpi=600)
    ax  = fig.add_subplot(111)
    ax.plot(q, r, color='black')
    ax.set_xlabel('$\mathregular{Q\ (Å^{-1})}$', fontsize=11, weight='bold')
    ax.set_ylabel('Reflectivity (arb.)', fontsize=11, weight='bold')
    ax.set_yscale('log')
    return fig, ax

def save_plot(fig: plt.Figure, save_path: str, file_name: str) -> None:
    """Saves a figure to a given directory.

    Args:
        fig (matplotlib.pyplot.Figure): figure to save.
        save_path (str): path to directory to save the figure to.
        file_name (str): name of file to save the plot as.

    """
    # Create the directory if not present.
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_path = os.path.join(save_path, file_name+'.png')
    fig.savefig(file_path, dpi=600)

if __name__ == '__main__':
    from structures import SymmetricBilayer

    # Plot each (fitted) objective on the same plot and save it.
    bilayer = SymmetricBilayer()

    fig = plt.figure(figsize=[9,7], dpi=600)
    ax = fig.add_subplot(111)

    #Plot Si-D2O
    model, data = bilayer.models[0], bilayer.datasets[0]
    ax.plot(data.x, model(data.x), color="red", zorder=20)
    ax.errorbar(data.x, data.y, data.y_err,
                color=None, label="Si/D2O Interface", marker="o", ms=3, lw=0,
                elinewidth=1, capsize=1.5)

    #Plot Si-DMPC-D2O
    model, data = bilayer.models[1], bilayer.datasets[1]
    offset = 1e-2
    ax.plot(data.x, model(data.x)*offset, color="red", zorder=20)
    ax.errorbar(data.x, data.y*offset, data.y_err*offset,
                label="DMPC Bilayer in D2O $\mathregular{(x10^{-2})}$",
                marker="o", ms=3, lw=0, elinewidth=1, capsize=1.5)

    #Plot Si-DMPC-H2O
    model, data = bilayer.models[2], bilayer.datasets[2]
    offset = 1e-3
    ax.plot(data.x, model(data.x)*offset, color="red", zorder=20)
    ax.errorbar(data.x, data.y*offset, data.y_err*offset,
                label="DMPC Bilayer in H2O $\mathregular{(x10^{-3})}$",
                marker="o", ms=3, lw=0, elinewidth=1, capsize=1.5)

    ax.set_xlabel("$\mathregular{Q\ (Å^{-1})}$", fontsize=11, weight='bold')
    ax.set_ylabel('Reflectivity (arb.)', fontsize=11, weight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1e-9, 2)
    ax.legend(loc='lower left')

    # Save the figure 4 plot.
    save_path = os.path.join('./results', str(bilayer))
    save_plot(fig, save_path, 'fitted_reflectivity_paper')
