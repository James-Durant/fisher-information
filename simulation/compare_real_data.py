import os
import numpy as np
import matplotlib.pyplot as plt

from refnx.dataset  import ReflectDataset
from refnx.reflect  import SLD, ReflectModel
from refnx.analysis import Objective, CurveFitter

from dynesty import NestedSampler
from dynesty import plotting as dyplot
from dynesty import utils    as dyfunc

from simulate import simulate_noisy, plot_objective

def QCS_sample():
    air       = SLD(0, name="Air")
    layer1    = SLD(1.795,  name="Layer 1 - Si")(thick=790.7,   rough=24.5)
    layer2    = SLD(6.385,  name="Layer 2 - Cu")(thick=297.9,   rough=3.5)
    substrate = SLD(3.354,  name="Substrate - Quartz")(thick=0, rough=12.9)
    return air | layer1 | layer2 | substrate

def plot_real_angles(data_path, files, save_path):
    ax = plt.figure(figsize=[9,7]).add_subplot(111)
    ax.set_xlim(0, 0.18)
    ax.set_ylim(1e-7, 2)
    ax.set_xlabel("$\mathregular{Q\ (Å^{-1})}$", fontsize=11, weight='bold')
    ax.set_ylabel('Reflectivity (arb.)',         fontsize=11, weight='bold')
    ax.set_yscale('log')

    for file_name in files:
        data = np.loadtxt(data_path+file_name, delimiter='    ')
        ax.errorbar(data[:,0], data[:,1], data[:,2], marker="o", ms=3, lw=0,
                    elinewidth=1, capsize=1.5, label=file_name)
    plt.legend()
    plt.savefig(save_path+"/real_reflectivity.png", dpi=600)

def plot_simulated_angles(angles, save_path):
    ax = plt.figure(figsize=[9,7]).add_subplot(111)
    ax.set_xlim(0, 0.18)
    ax.set_ylim(1e-7, 2)
    ax.set_xlabel("$\mathregular{Q\ (Å^{-1})}$", fontsize=11, weight='bold')
    ax.set_ylabel('Reflectivity (arb.)',         fontsize=11, weight='bold')
    ax.set_yscale('log')

    #Define the model for the real dataset.
    structure = QCS_sample()

    #Simulate noisy datasets with different angles corresponding to the real data.
    for angle in angles:
        points, time = angles[angle]
        _, q, r, r_error, _ = simulate_noisy(structure, angle, points, time)

        #Plot Q values against reflectivity with associated reflectivity error bars.
        ax.errorbar(q, r, r_error, marker="o", ms=3, lw=0, elinewidth=1,
                    capsize=1.5, label="simulated_{}°".format(str(angle)))
    plt.legend()
    plt.savefig(save_path+"/simulated_reflectivity.png", dpi=600)

def plot_real_all(data_path, save_path):
    structure = QCS_sample()
    model     = ReflectModel(structure, dq=2.5, scale=1, bkg=8e-7)
    data      = ReflectDataset(data_path)
    data.scale(np.max(data.y))

    objective = Objective(model, data)
    fig = plot_objective(objective)
    fig.savefig(save_path+"/real_fitted.png", dpi=600)

class Sampler:
    def __init__(self, objective):
        self.objective = objective
        self.ndim = len(objective.varying_parameters())
        self.sampler = NestedSampler(self.logl, self.objective.prior_transform, self.ndim)

    def sample(self):
        self.sampler.run_nested()
        results = self.sampler.results

        #Calculate the parameter means.
        weights = np.exp(results.logwt - results.logz[-1])
        mean, _ = dyfunc.mean_and_cov(results.samples, weights)
        self.logl(mean) #Update objective to use mean parameter values.

        fig, _ = dyplot.cornerplot(results, color='blue', quantiles=None, show_titles=True, max_n_ticks=3, truths=np.zeros(self.ndim), truth_color='black')
        return fig

    def logl(self, x):
        for i, parameter in enumerate(self.objective.varying_parameters()):
            parameter.value = x[i] #Update the model with given parameter values.
        return self.objective.logl()

def sample_real_data(data_path):
    structure = QCS_sample()
    
    for component in structure.components[1:-1]:
        sld_bounds   = (component.sld.real.value*0.5, component.sld.real.value*1.5)
        thick_bounds = (component.thick.value*0.5,    component.thick.value*1.5)
        rough_bounds = (component.rough.value*0.5,    component.rough.value*1.5)
        component.sld.real.setp(vary=True, bounds=sld_bounds)
        component.thick.setp(vary=True,    bounds=thick_bounds)
        component.rough.setp(vary=True,    bounds=rough_bounds)
    
    model = ReflectModel(structure, dq=2.5, scale=1, bkg=8e-7)
    data  = ReflectDataset(data_path)
    data.scale(np.max(data.y))

    objective = Objective(model, data)
    fitter    = CurveFitter(objective)
    fitter.fit('differential_evolution')
    fitter.sample(400) #Burn-in period
    fitter.reset()
    fitter.sample(15, nthin=100) #Main sampling stage.
    plot_objective(objective)
    objective.corner()
    plt.show()
    
    sampler = Sampler(objective)
    sampler.sample()
    plt.show()
    plot_objective(objective)

if __name__ == "__main__":
    save_path = "./results"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data_path = "./real_data/"
    real_data = ["QCS_03_1uA.dat", "QCS_04_1uA.dat", "QCS_05_1uA.dat",
                 "QCS_06_1uA.dat", "QCS_07_1uA.dat", "QCS_20_1uA.dat"]
    sample_real_data(data_path+"QCS_all.dat")
    plot_real_all(data_path+"QCS_all.dat", save_path)
    plot_real_angles(data_path, real_data, save_path)

    angles = {0.3: (84, 1), #Angle: (Points, Time)
              0.4: (49, 1),
              0.5: (55, 1),
              0.6: (64, 1),
              0.7: (75, 1),
              2.0: (53, 1)
             }
    plot_simulated_angles(angles, save_path)
