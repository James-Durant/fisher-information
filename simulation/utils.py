import matplotlib.pyplot as plt
import numpy as np
import os

from refnx.reflect import ReflectModel

def plot_sld_profile(structure, colour='black', label=None):
    """Plots the SLD profile of a given `structure`.

    Args:
        structure (refnx.reflect.Structure): the sample to plot the SLD profile of.
        colour (string): colour to use for the SLD profile.
        label (string): label to use for the SLD profile.

    """
    fig = plt.figure(figsize=[9,7], dpi=600)
    ax  = fig.add_subplot(111)
    #Plot the SLD profile with or without a label.
    ax.plot(*structure.sld_profile(), color=colour, label=label)
    ax.set_xlabel("$\mathregular{Distance\ (\AA)}$", fontsize=11, weight='bold')
    ax.set_ylabel("$\mathregular{SLD\ (10^{-6} \AA^{-2})}$", fontsize=11, weight='bold')
    return fig, ax

def reflectivity_curve(structure, q_min=0.005, q_max=0.3, points=500, dq=2, bkg=1e-7):
    """Plots the model reflectivity curve of a given `structure`.

    Args:
        structure (refnx.reflect.Structure): the sample to plot the reflectivity curve of.
        q_min (float): minimum Q value for the plot.
        q_max (float): maximum Q value for the plot.
        points (int): number of reflectivity points to plot.
        dq (float): instrument resolution parameter.
        bkg (float): instrument background parameter.

    """
    model = ReflectModel(structure, dq=dq, scale=1, bkg=bkg) #Define a model.
    q = np.logspace(np.log10(q_min), np.log10(q_max), points)
    r = model(q) #Calculate the model reflectivity.

    #Plot the model reflectivity against Q.
    fig = plt.figure(figsize=[9,7], dpi=600)
    ax  = fig.add_subplot(111)
    ax.plot(q, r, color="black")
    ax.set_xlabel("$\mathregular{Q\ (Å^{-1})}$", fontsize=11, weight='bold')
    ax.set_ylabel('Reflectivity (arb.)',         fontsize=11, weight='bold')
    ax.set_yscale('log')
    return fig, ax

def save_plot(fig, save_path, plot_type):
    """Saves a figure to a given directory.

    Args:
        fig (matplotlib.pyplot.Figure): figure to save.
        save_path: path to directory to save the figure to.
        plot_type: filename for the saved figure.

    """
    if not os.path.exists(save_path): #Create the directory if not present.
        os.makedirs(save_path)
    fig.savefig(save_path+"/"+plot_type+".png", dpi=600)

def plot_objective(objective, show_fit=True, ylim=1e-7):
    """Plots the fit of a given `objective` against the objective's data.

    Args:
        objective (refnx.analysis.Objective): the objective to plot.
        show_fit (Boolean): whether to display the objective fit or not.
        ylim (float): minimum reflectivity value to plot.

    """
    fig = plt.figure(figsize=[9,7], dpi=600)
    ax = fig.add_subplot(111)

    #Add the data.
    q = objective.data.x
    r, r_error, r_model = objective._data_transform(model=objective.generative())
    ax.errorbar(q, r, r_error, color="black", marker="o", ms=3, lw=0, elinewidth=1, capsize=1.5)

    if show_fit: #Add the prediction/fit.
        ax.plot(q, r_model, color="red", zorder=20)

    ax.set_xlabel("$\mathregular{Q\ (Å^{-1})}$", fontsize=11, weight='bold')
    ax.set_ylabel('Reflectivity (arb.)',         fontsize=11, weight='bold')
    ax.set_yscale('log')
    ax.set_xlim(0, 0.3)
    ax.set_ylim(ylim, 2)
    return fig
