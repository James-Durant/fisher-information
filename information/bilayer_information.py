import os, sys
sys.path.append("../simulation") #Adds directory to Python modules path.

import matplotlib.pyplot as plt
import numpy as np

from refnx.dataset  import ReflectDataset
from refnx.reflect  import SLD
from refnx.analysis import Objective

from fisher   import calc_FIM
from simulate import simulate_single_contrast

def bilayer_contrast_sample(contrast):
    """Creates a lipid bilayer sample using given `contrast` SLD.

    Args:
        contrast (float): SLD of the contrast choice.

    Returns:
        refnx.reflect.Structure: the lipid bilayer sample.

    """
    substrate        = SLD( 2.047,                 name="Silicon Substrate")
    SiO2_layer       = SLD( 3.41,                  name="Silicon Oxide Layer")(thick=8,  rough=3)
    water_layer      = SLD( contrast,              name="Water Layer"        )(thick=6,  rough=3)
    inner_headgroups = SLD( 0.76*0.8+contrast*0.2, name="Inner Headgroups"   )(thick=10, rough=5)
    chain_region     = SLD(-0.46*0.9+contrast*0.1, name="Chain Region"       )(thick=25, rough=4)
    outer_headgroups = SLD( 0.76*0.5+contrast*0.5, name="Outer Headgroups"   )(thick=9,  rough=7)
    bulk_water       = SLD( contrast,              name="Bulk Water"         )(thick=0,  rough=4)
    return substrate | SiO2_layer | water_layer | inner_headgroups | chain_region | outer_headgroups | bulk_water

def calc_fisher(model, data):
    """Calculates the Fisher information for the thickness parameters of the given `model`.

    Args:
        model (refnx.reflect.ReflectModel): the model to calculate the Fisher information with.
        data (numpy.ndarray): reflectivity and corresponding flux data.

    Returns:
        numpy.ndarray: Fisher information values for each thickness parameter.
        list: parameter names for use in matplotlib legend.
    """
    #Iterate over each layer but skip substrate and bulk water.
    for component in model.structure.components[1:-1]:
        #Vary the thickness 50% above and below the ground truth.
        thick_bounds = (component.thick.value*0.5, component.thick.value*1.5)
        component.thick.setp(vary=True, bounds=thick_bounds)

    q, r, r_error, flux = data[:,0], data[:,1], data[:,2], data[:,3]
    objective = Objective(model, ReflectDataset([q, r, r_error]))
    xi = objective.varying_parameters()
    g  = calc_FIM(q, xi, flux, model)

    return np.diag(g), [param.name for param in xi]

def plot_contrast_information(contrasts, angle_times, bkg, dq, save_path):
    """Plots the Fisher information for each thickness of the bilayer sample for
       each contrast in a given array of `contrasts`.

    Args:
        contrasts (numpy.ndarray): array of contrast SLDs to calculate the Fisher information for.
        angle_times (dict): dictionary of number of points and measurement times for each angle.
        bkg (float): value of the instrument background.
        dq (foat): value for the instrument resolution.
        save_path (string): path to directory to save plot to.

    """
    information = []
    for contrast in contrasts:
        #Simulate a dataset for the bilayer sample with selected contrast SLD.
        structure   = bilayer_contrast_sample(contrast)
        model, data = simulate_single_contrast(structure, angle_times, dq=dq, bkg=bkg, bkg_rate=bkg)

        #Calculate the thickness Fisher information values.
        fisher_info, names = calc_fisher(model, data)
        information.append(fisher_info)

    information = np.array(information)
    fig = plt.figure(figsize=[9,7], dpi=600)
    ax  = fig.add_subplot(111)
    for i in range(len(names)): #Plot the Fisher information values for each thickness parameter.
        ax.plot(contrasts, information[:,i], label=names[i])

    ax.set_xlabel("$\mathregular{Contrast\ SLD\ (10^{-6} \AA^{-2})}$", fontsize=11, weight='bold')
    ax.set_ylabel('Fisher Information', fontsize=11, weight='bold')
    ax.legend()
    fig.savefig(save_path+"/thickness_information.png", dpi=600)

if __name__ == "__main__":
    save_path = "./results/bilayer_contrast_sample"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    angle_times = {0.7: (70, 5), #Angle: (Points, Time)
                   2.0: (70, 20)}
    bkg = 1e-7
    dq  = 3
    contrasts = np.arange(-0.56, 6.35, 0.1)
    plot_contrast_information(contrasts, angle_times, bkg, dq, save_path)
