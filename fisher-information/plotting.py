import matplotlib.pyplot as plt
import numpy as np
import os

from refnx.reflect import ReflectModel

def plot_sld_profile(structure, distances=None, colour='black', label=False):
    """Plots the SLD profile of a given structure.

    Args:
        structure (refnx.reflect.Structure): the structure to plot the SLD profile of.
        distances (numpy.ndarray): the range of values to use on the x-axis.
        colour (string): colour to use for the SLD profile.
        label (Boolean): whether to use the structure's name in the plot's legend.

    Returns:
        fig (matplotlib.pyplot.Figure): figure containing a plotted SLD profile.
        ax (matplotlib.pyplot.Axis): axis containing a plotted SLD profile.

    """
    fig = plt.figure(figsize=[9,7], dpi=600)
    ax = fig.add_subplot(111)

    # Plot the SLD profile with or without a label.
    ax.plot(*structure.sld_profile(distances), color=colour,
            label=structure.name if label else None)

    ax.set_xlabel('$\mathregular{Distance\ (\AA)}$', fontsize=11, weight='bold')
    ax.set_ylabel('$\mathregular{SLD\ (10^{-6} \AA^{-2})}$', fontsize=11, weight='bold')
    return fig, ax

def plot_sld_profiles(structures, distances=None, label=True):
    """Plots the SLD profiles of a given list structures.

    Args:
        structures (list): the structures to plot the SLD profiles of.
        distances (numpy.ndarray): the range of values to use on the x-axis.
        label (Boolean): whether to use the structures' name in the plot's legend.

    Returns:
        fig (matplotlib.pyplot.Figure): figure containing plotted SLD profiles.
        ax (matplotlib.pyplot.Axis): axis containing plotted SLD profiles.

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

def plot_objective(objective, colour='black', label=False):
    """Plots the fit of a given objective against the objective's data.

    Args:
        objective (refnx.analysis.Objective): the objective to plot.
        colour (string): colour to use for the objective's data points.
        label (Boolean): whether to use the structure's name in the plot's legend.

    Returns:
        fig (matplotlib.pyplot.Figure): figure containing a plotted objective.
        ax (matplotlib.pyplot.Axis): axis containing a plotted objective.

    """
    # Plot the reflectivity data.
    fig, ax = plot_refdata(objective.data, colour,
                           objective.model.structure.name if label else None)

    # Plot the fit.
    q = objective.data.x
    ax.plot(q, objective.model(q), color='red', zorder=20)

    return fig, ax

def plot_objectives(objectives, label=True):
    """Plots the fits of a given list of objectives against the objectives' data.

    Args:
        objectives (list): list of objectives to plot.
        label (Boolean): whether to include a legend in the plot.

    Returns:
        fig (matplotlib.pyplot.Figure): figure containing plotted objectives.
        ax (matplotlib.pyplot.Axis): axis containing plotted objectives.

    """
    # Get the figure and axis by plotting the first objective by itself.
    fig, ax = plot_objective(objectives[0], None, label)

    # Plot the rest of the objectives on the same axis.
    for objective in objectives[1:]:
        q, r, r_error = objective.data.x, objective.data.y, objective.data.y_err

        # Plot the reflectivity data.
        ax.errorbar(q, r, r_error, marker='o', ms=3, lw=0, elinewidth=1, capsize=1.5,
                    label=objective.model.structure.name if label else None)

        # Plot the fit.
        ax.plot(q, objective.model(q), color='red', zorder=20)

    if label: # If labelling, create the legend.
        ax.legend()

    return fig, ax

def plot_refdata(data, colour='black', label=None):
    """Plots a given reflectivity dataset.

    Args:
        data (refnx.dataset.ReflectDataset): the dataset to plot.
        colour (string): the colour to plot the data points using
        label (string): label of the dataset for inclusion in plot's legend.

    Returns:
        fig (matplotlib.pyplot.Figure): figure containing a plotted reflectivity dataset.
        ax (matplotlib.pyplot.Axis): axis containing a plotted reflectivity dataset.

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

def plot_reflectivity_curve(structure, q_min=0.005, q_max=0.3, points=500, dq=2, bkg=1e-7):
    """Plots the model reflectivity curve of a given structure.

    Args:
        structure (refnx.reflect.Structure): the structure to plot the reflectivity curve of.
        q_min (float): minimum Q value for the plot.
        q_max (float): maximum Q value for the plot.
        points (int): number of reflectivity points to plot.
        dq (float): instrument resolution parameter.
        bkg (float): instrument background parameter.

    Returns:
        fig (matplotlib.pyplot.Figure): figure containing a plotted reflectivity curve.
        ax (matplotlib.pyplot.Axis): axis containing a plotted reflectivity curve.

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

def save_plot(fig, save_path, file_name):
    """Saves a figure to a given directory.

    Args:
        fig (matplotlib.pyplot.Figure): the figure to save.
        save_path: path to the directory to save the figure to.
        file_name: name of the file to save the plot as.

    """
    # Create the directory if not present.
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_path = os.path.join(save_path, file_name+'.png')
    fig.savefig(file_path, dpi=600)
