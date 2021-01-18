import matplotlib.pyplot as plt
import numpy as np
import os, refnx

from refnx.dataset  import ReflectDataset
from refnx.reflect  import SLD, LipidLeaflet, ReflectModel
from refnx.analysis import Parameter, GlobalObjective, Objective, CurveFitter

from simulation.structures import sld_profile
from simulation.simulate   import simulate_single_contrast, plot_objective
from information.utils     import calc_FIM

def DMPC_sample_refnx():
    path = os.path.join(os.path.dirname(refnx.__file__), 'analysis', 'test')
    
    D2O_data   = ReflectDataset(os.path.join(path, 'c_PLP0016596.dat'))
    hdmix_data = ReflectDataset(os.path.join(path, 'c_PLP0016601.dat'))
    H2O_data   = ReflectDataset(os.path.join(path, 'c_PLP0016607.dat'))
    
    Si    = SLD(2.07)
    SiO2  = SLD(3.47)
    D2O   = SLD(6.36)
    hdmix = SLD(2.07)
    H2O   = SLD(-0.56)

    D2O.real.setp(vary=True, bounds=(6.1, 6.36))

    apm     = Parameter(56, 'area per molecule', vary=True, bounds=(52, 65))
    b_heads = Parameter(6.01e-4, 'b_heads')
    b_tails = Parameter(-2.92e-4, 'b_tails')
    v_heads = Parameter(319, 'v_heads')
    v_tails = Parameter(782, 'v_tails')
    
    inner_head_thickness = Parameter(9, 'inner_head_thickness', vary=True, bounds=(4, 11))
    outer_head_thickness = Parameter(9, 'outer_head_thickness', vary=True, bounds=(4, 11))
    tail_thickness       = Parameter(14, 'tail_thickness', vary=True, bounds=(10, 17))
    
    inner_leaflet = LipidLeaflet(apm,
                                 b_heads, v_heads, inner_head_thickness,
                                 b_tails, v_tails, tail_thickness,
                                 3, 3)
    outer_leaflet = LipidLeaflet(apm,
                                 b_heads, v_heads, outer_head_thickness,
                                 b_tails, v_tails, tail_thickness,
                                 3, 0, reverse_monolayer=True)
    
    SiO2_layer = SiO2(thick=15, rough=3)
    SiO2_layer.thick.setp(vary=True, bounds=(2, 30))
    SiO2_layer.rough.setp(vary=True, bounds=(0, 7))
    SiO2_layer.vfsolv.setp(0.1, vary=True, bounds=(0., 0.5))
    
    solv_roughness = Parameter(3, 'bilayer/solvent - rough')
    solv_roughness.setp(vary=True, bounds=(0, 5))
    
    D2O_structure   = Si | SiO2_layer | inner_leaflet | outer_leaflet | D2O(0,   solv_roughness)
    hdmix_structure = Si | SiO2_layer | inner_leaflet | outer_leaflet | hdmix(0, solv_roughness)
    H2O_structure   = Si | SiO2_layer | inner_leaflet | outer_leaflet | H2O(0,   solv_roughness)
    
    D2O_structure.name   = "D2O"
    hdmix_structure.name = "hdmix"
    H2O_structure.name   = "H2O"
    
    D2O_model   = ReflectModel(D2O_structure)
    hdmix_model = ReflectModel(hdmix_structure)
    H2O_model   = ReflectModel(H2O_structure)
    
    D2O_model.scale.setp(vary=True, bounds=(0.9, 1.1))
    D2O_model.bkg.setp(vary=True, bounds=(-1e-6, 1e-6))
    hdmix_model.bkg.setp(vary=True, bounds=(-1e-6, 1e-6))
    H2O_model.bkg.setp(vary=True, bounds=(-1e-6, 1e-6))
    
    D2O_objective   = Objective(D2O_model,   D2O_data)
    hdmix_objective = Objective(hdmix_model, hdmix_data)
    H2O_objective   = Objective(H2O_model,   H2O_data)

    return GlobalObjective([D2O_objective, hdmix_objective, H2O_objective])

def fit_objective(objective, save_path):
    fitter = CurveFitter(objective)
    fitter.fit('differential_evolution')
    
    with open(save_path+"/parameters.txt", "w") as file:
        file.write(str(objective))
    
    #Plot each objective fit against the corresponding dataset.
    for objective in objective.objectives:
        fig = plot_objective(objective, ylim=1e-8)
        fig.savefig(save_path+"/{}_fitted.png".format(objective.model.structure.name), dpi=600)

def plot_sld_profiles(objective, save_path):
    structures = [objective.model.structure for objective in objective.objectives]
    
    fig, ax = sld_profile(structures[0], colour=None, label=structures[0].name)
    for structure in structures[1:]:
        ax.plot(*structure.sld_profile(), label=structure.name) #Plot SLD profile.
    ax.legend()
    fig.savefig(save_path+"/sld_profiles.png", dpi=600)

def DMPC_sample_using(contrast):
    SiO2_layer    = SLD(3.47*(1-0.14397)+contrast*0.14397)(thick=12.5021, rough=4.60093)
    inner_leaflet = LipidLeaflet(57.0848, 0.000601, 319, 9.61555, -0.000292, 782, 13.7036, 3, 3)
    outer_leaflet = LipidLeaflet(57.0848, 0.000601, 319, 5.67973, -0.000292, 782, 13.7036, 3, 0, reverse_monolayer=True)
    
    SiO2_layer.thick.setp(vary=True)
    inner_leaflet.thickness_heads.setp(vary=True)
    outer_leaflet.thickness_heads.setp(vary=True)
    inner_leaflet.thickness_tails.setp(vary=True)
    outer_leaflet.thickness_tails.setp(vary=True)
    
    SiO2_layer.thick.name = "SiO2 - thick"
    inner_leaflet.thickness_heads.name = "Inner heads - thick"
    outer_leaflet.thickness_heads.name = "Outer heads - thick"
    inner_leaflet.thickness_tails.name = "Inner tails - thick"
    outer_leaflet.thickness_tails.name = "Outer tails - thick"
    
    return SLD(2.07) | SiO2_layer | inner_leaflet | outer_leaflet | SLD(contrast)(thick=0, rough=0.863971)

def calc_fisher(model, data):
    """Calculates the Fisher information for the thickness parameters of the given `model`.

    Args:
        model (refnx.reflect.ReflectModel): the model to calculate the Fisher information with.
        data (numpy.ndarray): reflectivity and corresponding flux data.

    Returns:
        numpy.ndarray: Fisher information values for each thickness parameter.
        list: parameter names for use in matplotlib legend.
    """
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
        structure   = DMPC_sample_using(contrast)
        model, data = simulate_single_contrast(structure, angle_times, dq=dq, bkg=bkg)

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

    objective = DMPC_sample_refnx()
    fit_objective(objective, save_path)
    plot_sld_profiles(objective, save_path)
    
    angle_times = {0.7: (70, 5), #Angle: (Points, Time)
                   2.0: (70, 20)}
    bkg = 5e-7
    dq  = 5
    contrasts = np.arange(-0.56, 5, 0.05)
    plot_contrast_information(contrasts, angle_times, bkg, dq, save_path)
