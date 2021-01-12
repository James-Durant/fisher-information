import os
import numpy as np
import matplotlib.pyplot as plt

from refnx.reflect import SLD, ReflectModel

def multiple_contrast_sample():
    air       = SLD(0,   name="Air")
    D2O       = SLD(6.3, name="D20")
    layer1    = SLD(2.0, name="Layer 1")(thick=50, rough=6)
    layer2    = SLD(1.7, name="Layer 2")(thick=15, rough=2)
    layer3    = SLD(0.8, name="Layer 3")(thick=60, rough=2)
    layer4    = SLD(3.2, name="Layer 4")(thick=40, rough=2)
    layer5    = SLD(4.0, name="Layer 5")(thick=18, rough=2)
    substrate = SLD(2.047, name="Substrate")(thick=0, rough=2)

    structure_air = substrate | layer1 | layer2 | layer3 | layer4 | layer5 | air
    structure_D2O = substrate | layer1 | layer2 | layer3 | layer4 | layer5 | D2O
    return structure_air, structure_D2O

def easy_sample():
    air       = SLD(0, name="Air")
    layer1    = SLD(4, name="Layer 1")(thick=100, rough=2)
    layer2    = SLD(8, name="Layer 2")(thick=150, rough=2)
    substrate = SLD(2.047, name="Substrate")(thick=0, rough=2)
    structure = air | layer1 | layer2 | substrate
    return [structure]

def thin_layer_sample_1():
    air       = SLD(0, name="Air")
    layer1    = SLD(4, name="Layer 1")(thick=200, rough=2)
    layer2    = SLD(6, name="Layer 2")(thick=6,   rough=2)
    substrate = SLD(2.047, name="Substrate")(thick=0, rough=2)
    structure = air | layer1 | layer2 | substrate
    return [structure]

def thin_layer_sample_2():
    air       = SLD(0, name="Air")
    layer1    = SLD(4, name="Layer 1")(thick=200, rough=2)
    layer2    = SLD(5, name="Layer 2")(thick=30,  rough=6)
    layer3    = SLD(6, name="Layer 3")(thick=6,   rough=2)
    substrate = SLD(2.047, name="Substrate")(thick=0, rough=2)
    structure = air | layer1 | layer2 | layer3 | substrate
    return [structure]

def similar_sld_sample_1():
    air       = SLD(0,   name="Air")
    layer1    = SLD(0.9, name="Layer 1")(thick=80, rough=2)
    layer2    = SLD(1.0, name="Layer 2")(thick=50, rough=6)
    substrate = SLD(2.047, name="Substrate")(thick=0, rough=2)
    structure = air | layer1 | layer2 | substrate
    return [structure]

def similar_sld_sample_2():
    air       = SLD(0,   name="Air")
    layer1    = SLD(3.0, name="Layer 1")(thick=50, rough=2)
    layer2    = SLD(5.5, name="Layer 2")(thick=30, rough=6)
    layer3    = SLD(6.0, name="Layer 3")(thick=35, rough=2)
    substrate = SLD(2.047, name="Substrate")(thick=0, rough=2)
    structure = air | layer1 | layer2 | layer3 | substrate
    return [structure]

def many_param_sample():
    air       = SLD(0,   name="Air")
    layer1    = SLD(2.0, name="Layer 1")(thick=50, rough=6)
    layer2    = SLD(1.7, name="Layer 2")(thick=15, rough=2)
    layer3    = SLD(0.8, name="Layer 3")(thick=60, rough=2)
    layer4    = SLD(3.2, name="Layer 4")(thick=40, rough=2)
    layer5    = SLD(4.0, name="Layer 5")(thick=18, rough=2)
    substrate = SLD(2.047, name="Substrate")(thick=0, rough=2)
    structure = air | layer1 | layer2 | layer3 | layer4 | layer5 | substrate
    return [structure]

def plot_sld_profiles(save_path):
    """Plots the SLD profiles of all structures in this file.

    Args:
        save_path (string): path to the directory to save the SLD profiles to.

    """
    #Plot and save SLD profiles for each single contrast structure.
    for structure_func in [similar_sld_sample_1, similar_sld_sample_2,
                           thin_layer_sample_1,  thin_layer_sample_2,
                           easy_sample,          many_param_sample]:
        fig, _ = sld_profile(structure_func()[0])
        save_plot(fig, save_path+structure_func.__name__, "sld_profile")

    #Plot the two contrasts on the same axis.
    structure1, structure2 = multiple_contrast_sample()
    fig, ax = sld_profile(structure1, colour="C0", label="contrast_1")
    ax.plot(*structure2.sld_profile(), color="C1", label="contrast_2")
    ax.legend()
    save_plot(fig, save_path+"multiple_contrast_sample", "sld_profile")

def plot_reflectivity_curves(save_path):
    """Plots the reflectivity curves of all structures in this file.

    Args:
        save_path (string): path to the directory to save the reflectivity curves to.

    """
    #Plot and save reflectivity curves for each single contrast structure.
    for structure_func in [similar_sld_sample_1, similar_sld_sample_2,
                           thin_layer_sample_1,  thin_layer_sample_2,
                           easy_sample,          many_param_sample]:
        fig = reflectivity_curve(structure_func()[0])
        save_plot(fig, save_path+structure_func.__name__, "reflectivity")

    #Plot and save reflectivity curves for the multiple contrast sample.
    structure1, structure2 = multiple_contrast_sample()
    fig1 = reflectivity_curve(structure1)
    fig2 = reflectivity_curve(structure2)
    save_plot(fig1, save_path+"multiple_contrast_sample", "reflectivity_contrast_1")
    save_plot(fig2, save_path+"multiple_contrast_sample", "reflectivity_contrast_2")

def sld_profile(structure, colour='black', label=None):
    """Plots an SLD profile for a given `structure`.

    Args:
        structure (refnx.reflect.Structure): the sample to plot the SLD profile of.
        colour (string): colour to use for the SLD profile plot.
        label (string): label to use for the SLD profile of the structure.

    """
    fig = plt.figure(figsize=[9,7], dpi=600)
    ax  = fig.add_subplot(111)
    #Plot the SLD profile with or without a label.
    if label:
        ax.plot(*structure.sld_profile(), color=colour, label=label)
    else:
        ax.plot(*structure.sld_profile(), color=colour) #Plot SLD profile.
    ax.set_xlabel("$\mathregular{Distance\ (\AA)}$", fontsize=11, weight='bold')
    ax.set_ylabel("$\mathregular{SLD\ (10^{-6} \AA^{-2})}$", fontsize=11, weight='bold')
    return fig, ax

def reflectivity_curve(structure, q_min=0.005, q_max=0.3, points=500, dq=2, bkg=1e-7):
    """Plots the model reflectivity curve for a given `structure`.

    Args:
        structure (refnx.reflect.Structure): the sample to plot the reflectivity curve of.
        q_min (float): minimum Q value for plot.
        q_max (float): maximum Q value for plot.
        points (int): number of reflectivity points to plot.
        dq (float): instrument resolution parameter.
        bkg (float): instrument background parameter.

    """
    model = ReflectModel(structure, dq=dq, scale=1, bkg=bkg) #Define a model.
    q = np.logspace(np.log10(q_min), np.log10(q_max), points)
    r = model(q) #Calculate the model reflectivity.

    fig = plt.figure(figsize=[9,7], dpi=600)
    ax  = fig.add_subplot(111)
    ax.plot(q, r, color="black")
    ax.set_xlabel("$\mathregular{Q\ (Å^{-1})}$", fontsize=11, weight='bold')
    ax.set_ylabel('Reflectivity (arb.)',         fontsize=11, weight='bold')
    ax.set_yscale('log')
    return fig

def save_plot(fig, save_path, plot_type):
    """Saves a figure to a given directory.

    Args:
        fig (matplotlib.pyplot.Figure): figure to save.
        save_path: path to directory to save the figure to.
        plot_type: filename for the saved figure image.

    """
    if not os.path.exists(save_path): #Create the directory if not present.
        os.makedirs(save_path)
    fig.savefig(save_path+"/"+plot_type+".png", dpi=600)

if __name__ == "__main__":
    save_path = "../comparison/results/"
    plot_sld_profiles(save_path)
    plot_reflectivity_curves(save_path)
