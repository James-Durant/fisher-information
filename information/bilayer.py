import matplotlib.pyplot as plt
import numpy as np
import os

from refnx.dataset  import ReflectDataset
from refnx.reflect  import SLD, ReflectModel
from refnx.analysis import GlobalObjective, Objective, Parameter

from simulation.simulate import simulate_single_contrast
from simulation.utils    import plot_sld_profile, plot_objective
from information.utils   import calc_FIM

def DMPC_sample():
    """Constructs three structures for three contrasts of a DMPC sample:
       Si-SiO2-D2O, Si-SiO2-DMPC-D2O and Si-SiO2-DMPC-H2O. Data and fitted
       values are from the ISIS neutron training course.

    """
    #Define the known SLDs, roughnesses, thicknesses, hydrations, volumes and DMPC APM.
    substrate_sld     = 2.073
    substrate_rough   = 2
    sio2_sld          = 3.41
    sio2_thick        = 14.7073
    sio2_rough        = 2
    sio2_hydration    = 0.245273
    HG_bounds_waters  = 3.588
    bilayer_rough     = 6.5693
    bilayer_hydration = 0.073696
    DMPC_apm          = 49.8936
    DMPC_HG_vol       = 320.9
    DMPC_tails_vol    = 783.3
    water_vol         = 30.4

    #Known scattering lengths in Angstrom
    DMPC_HG_SL    =  6.41e-4
    DMPC_tails_SL = -3.08e-4
    H2O_SL        = -1.64e-5
    D2O_SL        =  2.00e-4

    #The two measured contrasts.
    D2O = SLD( 6.190)
    H2O = SLD(-0.5227)

    substrate = SLD(substrate_sld)

    sio2 = SLD(sio2_sld)(thick=sio2_thick, rough=substrate_rough)
    sio2.vfsolv.setp(sio2_hydration)

    #Relate HG bound waters to SL and volume
    headgroup_water_D2O_SL = HG_bounds_waters * D2O_SL
    Headgroup_water_H2O_SL = HG_bounds_waters * H2O_SL
    Headgroup_water_vol = HG_bounds_waters * water_vol

    #Add that to the HG volumes and SLs in both contrast cases
    vol_HG = DMPC_HG_vol + Headgroup_water_vol

    DMPC_HG_SL_D2O = DMPC_HG_SL + headgroup_water_D2O_SL
    DMPC_HG_SL_H2O = DMPC_HG_SL + Headgroup_water_H2O_SL

    #Calculate the SLD of the HG in both contrast cases
    SLD_HG_D2O = DMPC_HG_SL_D2O / vol_HG * 1e6 #SLD = sum b / v
    SLD_HG_H2O = DMPC_HG_SL_H2O / vol_HG * 1e6

    #Calculate the thickness from the HG volume over the lipid Area per molecule
    HG_thick = vol_HG / DMPC_apm #Thickness = v/APM
    #Calculate the SLD of the tails
    SLD_tails = DMPC_tails_SL / DMPC_tails_vol * 1e6
    #Calculate the thickness of the tails
    tails_thick = DMPC_tails_vol / DMPC_apm

    D2O_inner_HG = SLD(SLD_HG_D2O)(HG_thick, sio2_rough)
    D2O_outer_HG = SLD(SLD_HG_D2O)(HG_thick, bilayer_rough)

    H2O_inner_HG = SLD(SLD_HG_H2O)(HG_thick, sio2_rough)
    H2O_outer_HG = SLD(SLD_HG_H2O)(HG_thick, bilayer_rough)

    tails = SLD(SLD_tails)(tails_thick, bilayer_rough)

    #Set the DMPC hydradtion.
    D2O_inner_HG.vfsolv.setp(bilayer_hydration)
    D2O_outer_HG.vfsolv.setp(bilayer_hydration)
    H2O_inner_HG.vfsolv.setp(bilayer_hydration)
    H2O_outer_HG.vfsolv.setp(bilayer_hydration)
    tails.vfsolv.setp(bilayer_hydration)

    si_D2O_structure      = substrate | sio2 | D2O(rough=sio2_rough)
    si_DMPC_D2O_structure = substrate | sio2 | D2O_inner_HG | tails | tails | D2O_outer_HG | D2O(rough=bilayer_rough)
    si_DMPC_H2O_structure = substrate | sio2 | H2O_inner_HG | tails | tails | H2O_outer_HG | H2O(rough=bilayer_rough)

    si_D2O_structure.name      = "Si-D2O"
    si_DMPC_D2O_structure.name = "Si-DMPC-D2O"
    si_DMPC_H2O_structure.name = "Si-DMPC-H2O"

    return si_D2O_structure, si_DMPC_D2O_structure, si_DMPC_H2O_structure

def plot_sld_profiles(structures, save_path):
    """Plots the SLD profiles of each given contrast structure.

    Args:
        structures (list): the structures to plot the SLD profiles of.
        save_path (string): path to directory to save SLD profiles to.

    """
    distances = np.linspace(-20, 115, 500)
    #Plot the SLD profiles of each contrast on the same axis and save the plot.
    fig, ax = plot_sld_profile(structures[0], colour=None, label=structures[0].name, distance=distances)
    for structure in structures[1:]:
        ax.plot(*structure.sld_profile(distances), label=structure.name) #Plot SLD profile.
    ax.legend()
    fig.savefig(save_path+"/sld_profiles.png", dpi=600)

def plot_objectives(structures, save_path):
    """Plots the fitted reflectivity curves of each given contrast.

    Args:
        structures (list): the structures to plot the reflectivity curves of.
        save_path (string): path to directory to save curves to.

    """
    si_D2O_structure, si_DMPC_D2O_structure, si_DMPC_H2O_structure = structures

    #Define a model for each structure using known resolution, scale and backgrounds.
    si_D2O_model = ReflectModel(si_D2O_structure, dq=2, scale=0.677763, bkg=3.20559e-06)
    si_DMPC_D2O_model = ReflectModel(si_DMPC_D2O_structure, dq=2, scale=0.645217, bkg=2.05875e-06)
    si_DMPC_H2O_model = ReflectModel(si_DMPC_H2O_structure, dq=2, scale=0.667776, bkg=2.80358e-06)

    #Load the data for each measured contrast.
    si_D2O_data = ReflectDataset("./data/{}.dat".format(si_D2O_structure.name))
    si_DMPC_D2O_data = ReflectDataset("./data/{}.dat".format(si_DMPC_D2O_structure.name))
    si_DMPC_H2O_data = ReflectDataset("./data/{}.dat".format(si_DMPC_H2O_structure.name))

    #Define objectives using the models and loaded data.
    si_D2O_objective = Objective(si_D2O_model, si_D2O_data)
    si_DMPC_D2O_objective = Objective(si_DMPC_D2O_model, si_DMPC_D2O_data)
    si_DMPC_H2O_objective = Objective(si_DMPC_H2O_model, si_DMPC_H2O_data)

    global_objective = GlobalObjective([si_D2O_objective, si_DMPC_D2O_objective, si_DMPC_H2O_objective])

    #Save the parameters to a .txt file.
    with open(save_path+"/parameters.txt", "w") as file:
        file.write(str(global_objective))

    #Plot each (fitted) objective and save the plot.
    for objective in global_objective.objectives:
        fig = plot_objective(objective)
        fig.savefig(save_path+"/{}_reflectivity.png".format(objective.model.structure.name), dpi=600)

def DMPC_using_contrast(contrast_sld, vary_thick=True, vary_rough=True):
    """Creates a structure representing the DMPC sample above but measured
       using a given bulk contrast.

    Args:
        contrast_sld (float): the SLD of the contrast to measure.
        vary_thick (Boolean): whether to vary the thickness parameters.
        vary_rough (Boolean): whether to vary the roughness parameters.

    """
    #Define the same DMPC model as above but with variable contrast SLD.
    substrate_sld     = 2.073
    substrate_rough   = Parameter(2, vary=vary_rough, name="Substrate Roughness")
    sio2_sld          = 3.41
    sio2_thick        = Parameter(14.7073, vary=vary_thick, name="SiO2 Thickness")
    sio2_rough        = Parameter(2, vary=vary_rough, name="SiO2 Roughness")
    sio2_hydration    = 0.245273
    HG_bounds_waters  = 3.588
    bilayer_rough     = Parameter(6.5693, vary=vary_rough, name="Bilayer Roughness")
    bilayer_hydration = 0.073696
    DMPC_apm          = 49.8936
    DMPC_HG_vol       = 320.9
    DMPC_tails_vol    = 783.3
    water_vol         = 30.4
    DMPC_tails_SL     = -3.08e-4

    substrate = SLD(substrate_sld)

    sio2 = SLD(sio2_sld)(thick=sio2_thick, rough=substrate_rough)
    sio2.vfsolv.setp(sio2_hydration)

    vol_HG   = DMPC_HG_vol + HG_bounds_waters*water_vol
    HG_sld   = contrast_sld*0.27 + 1.98*0.73
    HG_thick = Parameter(vol_HG / DMPC_apm, vary=vary_thick, name="Headgroup Thickness")

    tails_sld   = DMPC_tails_SL / DMPC_tails_vol * 1e6
    tails_thick = Parameter(DMPC_tails_vol / DMPC_apm, vary=vary_thick, name="Tails Thickness")

    substrate = SLD(substrate_sld)
    sio2      = SLD(sio2_sld)(sio2_thick, substrate_rough)
    inner_HG  = SLD(HG_sld)(HG_thick, sio2_rough)
    outer_HG  = SLD(HG_sld)(HG_thick, bilayer_rough)
    tails     = SLD(tails_sld)(tails_thick, bilayer_rough)
    solution  = SLD(contrast_sld)(rough=bilayer_rough)

    sio2.vfsolv.setp(sio2_hydration)
    inner_HG.vfsolv.setp(bilayer_hydration)
    outer_HG.vfsolv.setp(bilayer_hydration)
    tails.vfsolv.setp(bilayer_hydration)

    return substrate | sio2 | inner_HG | tails | tails | outer_HG | solution

def plot_FIM(contrasts, angle_times, save_path):
    """Plots the FIM for each parameter of the DMPC sample for each contrast
       SLD in a given array of `contrasts`.

    Args:
        contrasts (numpy.ndarray): array of contrast SLDs to calculate the FIM with.
        angle_times (dict): dictionary of points and measurement times for each angle.
        save_path (string): path to directory to save FIM thickness plot to.

    """
    information = []
    for contrast_sld in contrasts:
        #Simulate data for the given sample with current contrast SLD.
        structure   = DMPC_using_contrast(contrast_sld)
        model, data = simulate_single_contrast(structure, angle_times)

        q, r, r_error, counts = data[:,0], data[:,1], data[:,2], data[:,3]

        #Calculate the FIM matrix and record the diagonal.
        objective = Objective(model, ReflectDataset([q, r, r_error]))
        xi = objective.varying_parameters()
        g  = calc_FIM(q, xi, counts, model)
        information.append(np.diag(g))

    #Plot the FIM values for each varied parameter.
    information = np.array(information)
    names = [param.name for param in xi]
    fig = plt.figure(figsize=[9,7], dpi=600)
    ax  = fig.add_subplot(111)
    for i in range(len(names)):
        ax.plot(contrasts, information[:,i], label=names[i])

    ax.set_xlabel("$\mathregular{Contrast\ SLD\ (10^{-6} \AA^{-2})}$", fontsize=11, weight='bold')
    ax.set_ylabel('Fisher Information', fontsize=11, weight='bold')
    ax.legend()
    fig.savefig(save_path+"/FIM_linear.png", dpi=600) #Save the plot.
    ax.set_yscale('log')
    fig.savefig(save_path+"/FIM_log.png", dpi=600)

if __name__ == "__main__":
    save_path = "./results/DMPC_sample"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    structures = DMPC_sample()
    plot_sld_profiles(structures, save_path)
    plot_objectives(structures, save_path)

    angle_times = {0.7: (70, 5), #Angle: (Points, Time)
                   2.0: (70, 20)}
    contrasts = np.arange(-0.56, 6.35, 0.01)
    plot_FIM(contrasts, angle_times, save_path)
