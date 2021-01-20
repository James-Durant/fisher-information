import matplotlib.pyplot as plt
import numpy as np
import os, refnx

from refnx.dataset  import ReflectDataset
from refnx.reflect  import SLD, LipidLeaflet, ReflectModel
from refnx.analysis import Parameter, GlobalObjective, Objective, CurveFitter

from simulation.simulate import simulate_single_contrast
from simulation.utils  import sld_profile, plot_objective
from information.utils import calc_FIM

def fit_DMPC_refnx_sample(save_path):
    """Fits a data set of three contrasts of a DMPC (1,2-dimyristoyl-sn-glycero-3-phosphocholine)
       bilayer measured at the solid-liquid interface.
       This is taken directly from the "Analysing lipid membrane data" refnx docs page.

    Args:
        save_path (string): path to directory to save fitted parameters to.

    """
    #Load the D2O, hdmix and H2O contrast data set that comes with refnx.
    path = os.path.join(os.path.dirname(refnx.__file__), 'analysis', 'test')
    D2O_data   = ReflectDataset(os.path.join(path, 'c_PLP0016596.dat'))
    hdmix_data = ReflectDataset(os.path.join(path, 'c_PLP0016601.dat'))
    H2O_data   = ReflectDataset(os.path.join(path, 'c_PLP0016607.dat'))

    #Define the solvent contrasts used.
    D2O   = SLD( 6.36, name="D2O")
    hdmix = SLD( 2.07, name="hdmix")
    H2O   = SLD(-0.56, name="H2O")

    D2O.real.setp(vary=True, bounds=(6.1, 6.36))

    #Define the area per molecule each DMPC molecule occupies at the surface.
    #We use the same area per molecule for the inner and outer leaflets.
    apm = Parameter(56, 'area per molecule', vary=True, bounds=(52, 65))

    #Sum of scattering lengths for the lipid head and tail in Angstrom.
    b_heads = Parameter(6.01e-4,  'b_heads')
    b_tails = Parameter(-2.92e-4, 'b_tails')

    #Volume occupied by the head and tail groups in cubic Angstrom.
    v_heads = Parameter(319, 'v_heads')
    v_tails = Parameter(782, 'v_tails')

    #Head and tail group thicknesses.
    inner_head_thickness = Parameter(9,  'inner_head_thickness', vary=True, bounds=(4, 11))
    outer_head_thickness = Parameter(9,  'outer_head_thickness', vary=True, bounds=(4, 11))
    tail_thickness       = Parameter(14, 'tail_thickness',       vary=True, bounds=(10, 17))

    #Inner and outer leaflets use the same area per molecule, same tail thickness, etc
    inner_leaflet = LipidLeaflet(apm,
                                 b_heads, v_heads, inner_head_thickness,
                                 b_tails, v_tails, tail_thickness,
                                 3, 3)
    #Reverse the monolayer for the outer leaflet because the tail groups face upwards.
    outer_leaflet = LipidLeaflet(apm,
                                 b_heads, v_heads, outer_head_thickness,
                                 b_tails, v_tails, tail_thickness,
                                 3, 0, reverse_monolayer=True)

    SiO2_layer = SLD(3.47)(thick=15, rough=3)
    SiO2_layer.thick.setp(vary=True, bounds=(2, 30))
    SiO2_layer.rough.setp(vary=True, bounds=(0, 7))
    SiO2_layer.vfsolv.setp(0.1, vary=True, bounds=(0, 0.5))

    solv_roughness = Parameter(3, 'bilayer/solvent - rough', vary=True, bounds=(0, 5))

    D2O_structure   = SLD(2.07) | SiO2_layer | inner_leaflet | outer_leaflet | D2O(0,   solv_roughness)
    hdmix_structure = SLD(2.07) | SiO2_layer | inner_leaflet | outer_leaflet | hdmix(0, solv_roughness)
    H2O_structure   = SLD(2.07) | SiO2_layer | inner_leaflet | outer_leaflet | H2O(0,   solv_roughness)

    D2O_structure.name   = "D2O"
    hdmix_structure.name = "hdmix"
    H2O_structure.name   = "H2O"

    D2O_model   = ReflectModel(D2O_structure)
    hdmix_model = ReflectModel(hdmix_structure)
    H2O_model   = ReflectModel(H2O_structure)

    D2O_model.scale.setp(vary=True, bounds=(0.9, 1.1))
    D2O_model.bkg.setp(vary=True,   bounds=(-1e-6, 1e-6))
    hdmix_model.bkg.setp(vary=True, bounds=(-1e-6, 1e-6))
    H2O_model.bkg.setp(vary=True,   bounds=(-1e-6, 1e-6))

    D2O_objective   = Objective(D2O_model,   D2O_data)
    hdmix_objective = Objective(hdmix_model, hdmix_data)
    H2O_objective   = Objective(H2O_model,   H2O_data)

    objective = GlobalObjective([D2O_objective, hdmix_objective, H2O_objective])

    save_path += "/DMPC_refnx_sample"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    fit_objective(objective, save_path)
    plot_sld_profiles(objective, save_path)

def fit_objective(objective, save_path):
    """Fits a given global objective and saves each individual contrast's fit.

    Args:
        save_path (string): path to directory to save fitted reflectivity curves to.

    """
    #Fit the objective with differential evolution.
    fitter = CurveFitter(objective)
    fitter.fit('differential_evolution')

    #Save the fitted parameters to a txt file.
    with open(save_path+"/parameters.txt", "w") as file:
        file.write(str(objective))

    #Plot each objective fit against the corresponding dataset and save the plot.
    for objective in objective.objectives:
        fig = plot_objective(objective, ylim=1e-8)
        fig.savefig(save_path+"/{}_fitted.png".format(objective.model.structure.name), dpi=600)

def plot_sld_profiles(objective, save_path):
    """Plots the SLD profiles of each contrast of a given global objective.

    Args:
        save_path (string): path to directory to save SLD profiles to.

    """
    structures = [objective.model.structure for objective in objective.objectives]

    #Plot the SLD profiles of each contrast on the same axis and save the plot.
    fig, ax = sld_profile(structures[0], colour=None, label=structures[0].name)
    for structure in structures[1:]:
        ax.plot(*structure.sld_profile(), label=structure.name) #Plot SLD profile.
    ax.legend()
    fig.savefig(save_path+"/sld_profiles.png", dpi=600)

def DMPC_refnx_sample(contrast):
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

    return SLD(2.047) | SiO2_layer | inner_leaflet | outer_leaflet | SLD(contrast)(thick=0, rough=0.863971)

def plot_FIM_thickness(sample, contrasts, angle_times, bkg, dq, save_path):
    """Plots the FIM for each thickness of the given `sample` for each contrast
       in a given array of `contrasts`.

    Args:
        sample (function): the sample to calculate the FIM with.
        contrasts (numpy.ndarray): array of contrast SLDs to calculate the FIM with.
        angle_times (dict): dictionary of points and measurement times for each angle.
        bkg (float): value of the instrument background.
        dq (foat): value for the instrument resolution.
        save_path (string): path to directory to save FIM thickness plot to.

    """
    information = []
    for contrast in contrasts:
        #Simulate data for the given sample with current contrast SLD.
        structure   = sample(contrast)
        model, data = simulate_single_contrast(structure, angle_times, dq=dq, bkg=bkg)

        q, r, r_error, counts = data[:,0], data[:,1], data[:,2], data[:,3]

        #Calculate the FIM matrix and record the diagonal.
        objective = Objective(model, ReflectDataset([q, r, r_error]))
        xi = objective.varying_parameters()
        g  = calc_FIM(q, xi, counts, model)
        information.append(np.diag(g))

    information = np.array(information)
    names = [param.name for param in xi]

    #Plot the FIM values for each thickness parameter.
    fig = plt.figure(figsize=[9,7], dpi=600)
    ax  = fig.add_subplot(111)
    for i in range(len(names)):
        ax.plot(contrasts, information[:,i], label=names[i])

    ax.set_xlabel("$\mathregular{Contrast\ SLD\ (10^{-6} \AA^{-2})}$", fontsize=11, weight='bold')
    ax.set_ylabel('Fisher Information', fontsize=11, weight='bold')
    ax.legend()
    fig.savefig(save_path+"/FIM_thickness.png", dpi=600) #Save the plot.

if __name__ == "__main__":
    save_path = "./results"

    fit_DMPC_refnx_sample(save_path)

    angle_times = {0.7: (70, 5), #Angle: (Points, Time)
                   2.0: (70, 20)}
    bkg = 1e-7
    dq  = 2
    contrasts = np.arange(-0.56, 6.35, 0.05)
    plot_FIM_thickness(DMPC_refnx_sample, contrasts, angle_times, bkg, dq, save_path+"/DMPC_refnx_sample")
