import os
import numpy as np
import matplotlib.pyplot as plt

from refnx.dataset  import ReflectDataset
from refnx.reflect  import SLD
from refnx.analysis import Parameter, GlobalObjective, Objective, CurveFitter

from simulate import simulate_multiple_contrasts

def bilayer_contrast_sample():
    """Creates a bilayer sample with three contrasts: H20, SMW (silicon-matched water) and D2O.

    Returns:
        refnx.reflect.Structure: three structures corresponding to each contrast.

    """
    #The SLDs of the three contrasts.
    H2O = -0.56
    SMW =  2.07
    D2O =  6.35

    substrate  = SLD(2.047, name="Silicon Substrate")
    SiO2_layer = SLD(3.41,  name="Silicon Oxide Layer")(thick=8, rough=3)

    name = "Water Layer"
    water_layer_thick = Parameter(6, name=name+" - Thick")
    water_layer_rough = Parameter(3, name=name+" - Rough")
    H2O_water_layer = SLD(H2O, name=name)(thick=water_layer_thick, rough=water_layer_rough)
    SMW_water_layer = SLD(SMW, name=name)(thick=water_layer_thick, rough=water_layer_rough)
    D2O_water_layer = SLD(D2O, name=name)(thick=water_layer_thick, rough=water_layer_rough)

    name = "Inner Headgroups"
    inner_headgroups_thick = Parameter(10, name=name+" - Thick")
    inner_headgroups_rough = Parameter(5,  name=name+" - Rough")
    H2O_inner_headgroups = SLD(0.76*0.8+H2O*0.2, name=name)(thick=inner_headgroups_thick, rough=inner_headgroups_rough)
    SMW_inner_headgroups = SLD(0.76*0.8+SMW*0.2, name=name)(thick=inner_headgroups_thick, rough=inner_headgroups_rough)
    D2O_inner_headgroups = SLD(0.76*0.8+D2O*0.2, name=name)(thick=inner_headgroups_thick, rough=inner_headgroups_rough)

    name = "Chain Region"
    chain_region_thick = Parameter(25, name=name+" - Thick")
    chain_region_rough = Parameter(4,  name=name+" - Rough")
    H2O_chain_region = SLD(-0.46*0.9+H2O*0.1, name=name)(thick=chain_region_thick, rough=chain_region_rough)
    SMW_chain_region = SLD(-0.46*0.9+SMW*0.1, name=name)(thick=chain_region_thick, rough=chain_region_rough)
    D2O_chain_region = SLD(-0.46*0.9+D2O*0.1, name=name)(thick=chain_region_thick, rough=chain_region_rough)

    name = "Outer Headgroups"
    outer_headgroups_thick = Parameter(9, name=name+" - Thick")
    outer_headgroups_rough = Parameter(7, name=name+" - Rough")
    H2O_outer_headgroups = SLD(0.76*0.5+H2O*0.5, name=name)(thick=outer_headgroups_thick, rough=outer_headgroups_rough)
    SMW_outer_headgroups = SLD(0.76*0.5+SMW*0.5, name=name)(thick=outer_headgroups_thick, rough=outer_headgroups_rough)
    D2O_outer_headgroups = SLD(0.76*0.5+D2O*0.5, name=name)(thick=outer_headgroups_thick, rough=outer_headgroups_rough)

    name = "Bulk Water"
    bulk_water_rough = Parameter(4, name=name+" - Rough")
    H2O_bulk_water = SLD(H2O, name=name)(rough=bulk_water_rough)
    SMW_bulk_water = SLD(SMW, name=name)(rough=bulk_water_rough)
    D2O_bulk_water = SLD(D2O, name=name)(rough=bulk_water_rough)

    H2O_structure = substrate | SiO2_layer | H2O_water_layer | H2O_inner_headgroups | H2O_chain_region | H2O_outer_headgroups | H2O_bulk_water
    SMW_structure = substrate | SiO2_layer | SMW_water_layer | SMW_inner_headgroups | SMW_chain_region | SMW_outer_headgroups | SMW_bulk_water
    D2O_structure = substrate | SiO2_layer | D2O_water_layer | D2O_inner_headgroups | D2O_chain_region | D2O_outer_headgroups | D2O_bulk_water

    H2O_structure.name = "H2O"
    SMW_structure.name = "SMW"
    D2O_structure.name = "D2O"

    #Vary the thickness and roughness of each layer.
    for param in [SiO2_layer.thick,       SiO2_layer.rough,
                  water_layer_thick,      water_layer_rough,
                  inner_headgroups_thick, inner_headgroups_rough,
                  chain_region_thick,     chain_region_rough,
                  outer_headgroups_thick, outer_headgroups_rough,
                                          bulk_water_rough]:
        #Set a 50% bound above and below the ground truth value.
        bounds = (param.value*0.5, param.value*1.5)
        param.setp(vary=True, bounds=bounds)
        #Initalise the parameter to some random value within its bounds.
        param.value = np.random.uniform(*bounds)

    return H2O_structure, SMW_structure, D2O_structure

def fit_models(models, datasets, save_path):
    """Fits the simulated data for each contrast and plots the result.

    Args:
        models (list): model of each contrast.
        datasets (list): experimental data for each contrast.
        save_path (string): path to directory to save the model fits to.

    """
    #Create a global objective to simultaneously fit the three contrasts.
    objective = GlobalObjective([Objective(model, ReflectDataset([data[:,0], data[:,1], data[:,2]]))
                                 for (model, data) in list(zip(models, datasets))])
    fitter = CurveFitter(objective)
    fitter.fit('differential_evolution')

    #Plot each objective fit against the corresponding dataset.
    for objective in objective.objectives:
        fig = plt.figure(figsize=[9,7], dpi=600)
        ax  = fig.add_subplot(111)

        # Add the data in a transformed fashion.
        y, y_err, model = objective._data_transform(model=objective.generative())
        ax.errorbar(objective.data.x, y, y_err, marker="o", ms=3, lw=0,  elinewidth=1,
                    capsize=1.5, label=objective.model.structure.name+" Simulated")
        #Add the fit
        ax.plot(objective.data.x, model, color="red", zorder=20, label="Fitted Model")

        ax.set_xlabel("$\mathregular{Q\ (Å^{-1})}$", fontsize=11, weight='bold')
        ax.set_ylabel('Reflectivity (arb.)',         fontsize=11, weight='bold')
        ax.set_xlim(0, 0.3)
        ax.set_yscale('log')
        ax.legend()
        fig.savefig(save_path+"/{}_reflectivity.png".format(objective.model.structure.name), dpi=600)

def plot_sld_profile(structures, save_path):
    """Plots the SLD profiles for a given list `structures`.

    Args:
        structures (list): list of refnx.reflect.Structure objects to plot the SLD profiles of.
        save_path (string): path to directory to save SLD profiles plot to.

    """
    fig = plt.figure(figsize=[9,7], dpi=600)
    ax  = fig.add_subplot(111)
    for structure in structures:
        ax.plot(*structure.sld_profile(), label=structure.name) #Plot SLD profile.

    ax.set_xlabel("$\mathregular{Distance\ (\AA)}$",         fontsize=11, weight='bold')
    ax.set_ylabel("$\mathregular{SLD\ (10^{-6} \AA^{-2})}$", fontsize=11, weight='bold')
    ax.legend()
    fig.savefig(save_path+"/sld_profiles.png", dpi=600)

def plot_models(models, save_path, q_min=0.005, q_max=0.3, points=300):
    """Plots the model reflectivity of each model in a given list of models.

    Args:
        models (list): list of models to plot the reflectivity of.
        save_path (string): path to directory to save model reflectivity plots to.
        q_min (float): minimum Q value to plot for each model.
        q_max (float): maximum Q value to plot for each model.
        points (int): number of reflectivity points to plot.

    """
    fig = plt.figure(figsize=[9,7], dpi=600)
    ax  = fig.add_subplot(111)
    #Plot over a equally log-spaced Q range.
    q = np.logspace(np.log10(q_min), np.log10(q_max), points)
    for model in models:
        ax.plot(q, model(q), label=model.structure.name) #Plot model reflectivity.

    ax.set_xlabel("$\mathregular{Q\ (Å^{-1})}$", fontsize=11, weight='bold')
    ax.set_ylabel('Reflectivity (arb.)',         fontsize=11, weight='bold')
    ax.set_yscale('log')
    ax.legend()
    fig.savefig(save_path+"/model_reflectivities.png", dpi=600)

if __name__ == "__main__":
    save_path = "./results/bilayer_contrast_sample"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    angle_times = {0.7: (70, 5), #Angle: (Points, Time)
                   2.0: (70, 20)}
    bkg = 1e-7
    dq  = 3

    structures = bilayer_contrast_sample()
    plot_sld_profile(structures, save_path)

    models, datasets  = simulate_multiple_contrasts(structures, angle_times, dq=dq, bkg=bkg, bkg_rate=bkg)
    plot_models(models, save_path)
    fit_models(models, datasets, save_path)
