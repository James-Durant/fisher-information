import matplotlib.pyplot as plt
import numpy as np

from refnx.dataset  import ReflectDataset
from refnx.reflect  import SLD, ReflectModel
from refnx.analysis import Parameter, GlobalObjective, Objective, CurveFitter

from simulate import simulate_noisy

def H2O_SMW_D2O_contrast_sample():
    H2O = -0.56
    SMW =  2.07
    D2O =  6.35
    
    substrate  = SLD(2.047, name="Silicon Substrate"  )(thick=0, rough=2)
    SiO2_layer = SLD(3.41,  name="Silicon Oxide Layer")(thick=8, rough=3)
    
    name = "Water Layer"
    water_layer_thick = Parameter(6, name=name+" - Thick")
    water_layer_rough = Parameter(5, name=name+" - Rough")
    H2O_water_layer = SLD(H2O, name=name)(thick=water_layer_thick, rough=water_layer_rough)
    SMW_water_layer = SLD(SMW, name=name)(thick=water_layer_thick, rough=water_layer_rough)
    D2O_water_layer = SLD(D2O, name=name)(thick=water_layer_thick, rough=water_layer_rough)
    
    name = "Inner Headgroups"
    inner_headgroups_thick = Parameter(10, name=name+" - Thick")
    inner_headgroups_rough = Parameter(4,  name=name+" - Rough")
    H2O_inner_headgroups = SLD(0.76*0.8+H2O*0.2, name=name)(thick=inner_headgroups_thick, rough=inner_headgroups_rough)
    SMW_inner_headgroups = SLD(0.76*0.8+SMW*0.2, name=name)(thick=inner_headgroups_thick, rough=inner_headgroups_rough)    
    D2O_inner_headgroups = SLD(0.76*0.8+D2O*0.2, name=name)(thick=inner_headgroups_thick, rough=inner_headgroups_rough)    
    
    name = "Chain Region"
    chain_region_thick = Parameter(25, name=name+" - Thick")
    chain_region_rough = Parameter(7,  name=name+" - Rough")
    H2O_chain_region = SLD(-0.46*0.9+H2O*0.1, name=name)(thick=chain_region_thick, rough=chain_region_rough)
    SMW_chain_region = SLD(-0.46*0.9+SMW*0.1, name=name)(thick=chain_region_thick, rough=chain_region_rough)
    D2O_chain_region = SLD(-0.46*0.9+D2O*0.1, name=name)(thick=chain_region_thick, rough=chain_region_rough)
    
    name = "Outer Headgroups"
    outer_headgroups_thick = Parameter(9, name=name+" - Thick")
    outer_headgroups_rough = Parameter(4, name=name+" - Rough")
    H2O_outer_headgroups = SLD(0.76*0.5+H2O*0.5, name=name)(thick=outer_headgroups_thick, rough=outer_headgroups_rough)
    SMW_outer_headgroups = SLD(0.76*0.5+SMW*0.5, name=name)(thick=outer_headgroups_thick, rough=outer_headgroups_rough)
    D2O_outer_headgroups = SLD(0.76*0.5+D2O*0.5, name=name)(thick=outer_headgroups_thick, rough=outer_headgroups_rough)

    name = "Bulk Water"
    H2O_bulk_water = SLD(H2O, name=name)
    SMW_bulk_water = SLD(SMW, name=name)
    D2O_bulk_water = SLD(D2O, name=name)
    
    H2O_structure = substrate | SiO2_layer | H2O_water_layer | H2O_inner_headgroups | H2O_chain_region | H2O_outer_headgroups | H2O_bulk_water
    SMW_structure = substrate | SiO2_layer | SMW_water_layer | SMW_inner_headgroups | SMW_chain_region | SMW_outer_headgroups | SMW_bulk_water
    D2O_structure = substrate | SiO2_layer | D2O_water_layer | D2O_inner_headgroups | D2O_chain_region | D2O_outer_headgroups | D2O_bulk_water
    
    H2O_structure.name = "H2O"
    SMW_structure.name = "SMW"
    D2O_structure.name = "D2O"
    
    for param in [water_layer_thick, water_layer_rough,  inner_headgroups_thick, inner_headgroups_rough,
                 chain_region_thick, chain_region_rough, outer_headgroups_thick, outer_headgroups_rough,
                 SiO2_layer.thick, SiO2_layer.rough]:
        param.setp(vary=True, bounds=(param.value*0.5, param.value*1.5))
    
    return H2O_structure, SMW_structure, D2O_structure

def simulate_experiment(structures, angle_times):
    datasets = []
    #For each contrast, simulate an experiment using angles with different measurement times.
    for structure in structures:
        q, r, r_error = [], [], []
        for angle in angle_times:
            #Simulate the experiment
            points, time = angle_times[angle]
            _, q_anlge, r_angle, r_error_angle, _ = simulate_noisy(structure, angle, points, time)
    
            #Combine the q, r and r_error values with the data of other angles.
            q += q_anlge
            r += r_angle
            r_error += r_error_angle
    
        data = np.zeros((len(angle_times)*points, 3))
        data[:,0] = q
        data[:,1] = r
        data[:,2] = r_error
        data = data[data[:,0].argsort()] #Sort by Q
        datasets.append(ReflectDataset([data[:,0], data[:,1], data[:,2]]))

    return datasets
    
def fit_models(models, datasets):
    objective = GlobalObjective([Objective(model, data) 
                                 for (model, data) in list(zip(models, datasets))])
    fitter = CurveFitter(objective)
    fitter.fit('differential_evolution')

    fig = plt.figure(figsize=[9,7], dpi=600)
    ax = fig.add_subplot(111)

    #Plot each objective fit against the corresponding dataset.
    for objective in objective.objectives:
        # Add the data in a transformed fashion.
        y, y_err, model = objective._data_transform(model=objective.generative())
        ax.errorbar(objective.data.x, y, y_err, marker="o", ms=3, lw=0,  elinewidth=1,
                    capsize=1.5, label=objective.model.structure.name+" Simulated")
        #Add the fit
        ax.plot(objective.data.x, model, color="red", zorder=20)

    plt.xlabel("$\mathregular{Q\ (Å^{-1})}$", fontsize=11, weight='bold')
    plt.ylabel('Reflectivity (arb.)',         fontsize=11, weight='bold')
    plt.xlim(0, 0.3)
    plt.yscale('log')
    plt.legend()
    plt.show()

def plot_sld_profile(structures):
    plt.figure(figsize=[9,7], dpi=600)
    for structure in structures:
        plt.plot(*structure.sld_profile(), label=structure.name) #Plot SLD profile.
    
    plt.xlabel("$\mathregular{Distance\ (\AA)}$",         fontsize=11, weight='bold')
    plt.ylabel("$\mathregular{SLD\ (10^{-6} \AA^{-2})}$", fontsize=11, weight='bold')
    plt.legend()
    plt.show()

def plot_models(models, q_min=0.005, q_max=0.3, points=300):
    plt.figure(figsize=[9,7], dpi=600).add_subplot(111)
    q = np.logspace(np.log10(q_min), np.log10(q_max), points)
    for model in models:
        plt.plot(q, model(q), label=model.structure.name)

    plt.xlabel("$\mathregular{Q\ (Å^{-1})}$", fontsize=11, weight='bold')
    plt.ylabel('Reflectivity (arb.)',         fontsize=11, weight='bold')
    plt.yscale('log')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    angle_times = {0.8: (50, 1), #Angle: (Points, Time)
                   3.2: (50, 1)} 
    bkg = 1e-7
    dq  = 3
    
    structures = H2O_SMW_D2O_contrast_sample()
    datasets   = simulate_experiment(structures, angle_times)
    plot_sld_profile(structures)
    
    models = [ReflectModel(structure, bkg=bkg, dq=dq) for structure in structures]
    plot_models(models)
    
    fit_models(models, datasets)
    
    