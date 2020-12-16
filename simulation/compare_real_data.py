import os
import numpy as np
import matplotlib.pyplot as plt

from refnx.dataset  import ReflectDataset
from refnx.reflect  import SLD, ReflectModel
from refnx.analysis import Objective, CurveFitter

from dynesty import NestedSampler
from dynesty import plotting as dyplot
from dynesty import utils    as dyfunc

from simulate import run_experiment, plot_objective

def QCS_sample():
    """Creates the QCS (Quartz, Copper, Silicon) sample for which the real data
       was recorded using.

    Returns:
        refnx.reflect.Structure: refnx representation of the real structure.

    """
    air       = SLD(0, name="Air")
    layer1    = SLD(1.795,  name="Layer 1 - Si")(thick=790.7,   rough=24.5)
    layer2    = SLD(6.385,  name="Layer 2 - Cu")(thick=297.9,   rough=3.5)
    substrate = SLD(3.354,  name="Substrate - Quartz")(thick=0, rough=12.9)
    return air | layer1 | layer2 | substrate

def plot_real_angles(data_path, files, save_path):
    """Creates a plot overlaying each dataset of each measured angle.

    Args:
        data_path (string): path to directory containing the real data.
        files (list): list of file names for each angle's dataset to plot.
        save_path (string): path to directory to save the plot to.

    """
    fig = plt.figure(figsize=[9,7], dpi=600)
    ax  = fig.add_subplot(111)
    ax.set_xlim(0, 0.18)
    ax.set_ylim(1e-7, 2)
    ax.set_xlabel("$\mathregular{Q\ (Å^{-1})}$", fontsize=11, weight='bold')
    ax.set_ylabel('Reflectivity (arb.)',         fontsize=11, weight='bold')
    ax.set_yscale('log')

    #For each measurement angle, load and plot the corresponding data.
    for file_name in files:
        data = np.loadtxt(data_path+file_name, delimiter='    ')
        #Plot Q values against reflectivity with associated reflectivity error bars.
        ax.errorbar(data[:,0], data[:,1], data[:,2], marker="o", ms=3, lw=0,
                    elinewidth=1, capsize=1.5, label=file_name)
    ax.legend()
    fig.savefig(save_path+"/real_reflectivity.png", dpi=600)

def plot_simulated_angles(angle_times, dq, bkg, save_path):
    """Simulates experiments for the QCS sample for a given set of angles.

    Args:
        angle_times (dict): dictionary of number of points and measurement times for each measured angle.
        dq (float): value to use for the model's instrument resolution parameter.
        bkg (float): value to use for the model's background parameter.
        save_path (string): directory to save the plot to.

    """
    fig = plt.figure(figsize=[9,7], dpi=600)
    ax  = fig.add_subplot(111)
    ax.set_xlim(0, 0.18) #Only plot up to the point that the real data goes up to.
    ax.set_ylim(1e-7, 2)
    ax.set_xlabel("$\mathregular{Q\ (Å^{-1})}$", fontsize=11, weight='bold')
    ax.set_ylabel('Reflectivity (arb.)',         fontsize=11, weight='bold')
    ax.set_yscale('log')

    #Define the model for the real dataset.
    structure = QCS_sample()

    #Simulate noisy datasets with different angles corresponding to the real data.
    for angle in angle_times:
        points, time = angle_times[angle]
        model = ReflectModel(structure, scale=1, dq=dq, bkg=bkg)
        q, r, r_error, _ = run_experiment(model, angle, points, time)

        #Plot Q values against reflectivity with associated reflectivity error bars.
        ax.errorbar(q, r, r_error, marker="o", ms=3, lw=0, elinewidth=1,
                    capsize=1.5, label="simulated_{}°".format(str(angle)))
    ax.legend()
    fig.savefig(save_path+"/simulated_reflectivity.png", dpi=600)

def plot_real_all(data_path, dq, bkg, save_path):
    """Creates a plot of all the angles combined (binned).

    Args:
        data_path (string): path to the directory containing the 'QCS_all.dat' file.
        dq (float): value to use for the model's instrument resolution parameter.
        bkg (float): value to use for the model's background parameter.
        save_path (string): path to the directory to save the plot to.

    """
    structure = QCS_sample()
    model     = ReflectModel(structure, scale=1, dq=dq, bkg=bkg)
    data      = ReflectDataset(data_path)
    data.scale(np.max(data.y)) #Scale the data so that the maximum reflectivity is 1.

    #Plot the fit of the real dataset and save it.
    objective = Objective(model, data)
    fig = plot_objective(objective)
    fig.savefig(save_path+"/real_fitted.png", dpi=600)

class Sampler:
    """The Sampler class contains the code nested sampling.

    Attributes:
        objective (refnx.analysis.Objective): the objective to fit.
        ndim (int): the dimensionality of the problem.
        sampler (dynesty.NestedSampler): a dynesty static nested sampler.

    """
    def __init__(self, objective):
        self.objective = objective
        self.ndim = len(objective.varying_parameters())
        self.sampler = NestedSampler(self.logl, self.objective.prior_transform, self.ndim)

    def sample(self):
        """Fits the given objective by sampling.

        Returns:
            matplotlib.pyplot.figure: corner plot figure from the sampling routine.

        """
        self.sampler.run_nested()
        results = self.sampler.results

        #Calculate the parameter means.
        weights = np.exp(results.logwt - results.logz[-1])
        mean, _ = dyfunc.mean_and_cov(results.samples, weights)
        self.logl(mean) #Update objective to use mean parameter values.

        fig, _ = dyplot.cornerplot(results, color='blue', quantiles=None, show_titles=True, max_n_ticks=3, truths=np.zeros(self.ndim), truth_color='black')
        return fig

    def logl(self, x):
        """Calculates the log-likelihood of the parameters `x` against the model.

        Args:
            x (numpy.ndarray): array of parameter values.

        Returns:
            float: log-likelihood of the parameters `x`.

        """
        for i, parameter in enumerate(self.objective.varying_parameters()):
            parameter.value = x[i] #Update the model with given parameter values.
        return self.objective.logl()

def sample_real_data(data_path, dq, bkg, save_path):
    """Samples the data in the 'QCS_all.dat' file using MCMC and nested sampling.

    Args:
        data_path (string): path to the directory containing the 'QCS_all.dat' file.
        dq (float): value to use for the model's instrument resolution parameter.
        bkg (float): value to use for the model's background parameter.
        save_path (string): path to the directory to save the sampling corner plots to.

    """
    structure = QCS_sample()

    #Vary the SLD, thickness and roughness of each layer (skip air and the substrate).
    for component in structure.components[1:-1]:
        sld_bounds   = (component.sld.real.value*0.5, component.sld.real.value*1.5)
        thick_bounds = (component.thick.value*0.5,    component.thick.value*1.5)
        rough_bounds = (component.rough.value*0.5,    component.rough.value*1.5)
        component.sld.real.setp(vary=True, bounds=sld_bounds)
        component.thick.setp(vary=True,    bounds=thick_bounds)
        component.rough.setp(vary=True,    bounds=rough_bounds)

    model = ReflectModel(structure, scale=1, dq=dq, bkg=bkg)
    data  = ReflectDataset(data_path)
    data.scale(np.max(data.y)) #Scale the data so that the maximum reflectivity is 1.

    #Fit using differential evolution then sample the result using MCMC sampling.
    objective = Objective(model, data)
    fitter    = CurveFitter(objective)
    fitter.fit('differential_evolution')
    fitter.sample(400) #Burn-in period.
    fitter.reset()
    fitter.sample(15, nthin=100) #Main sampling stage.
    fig = objective.corner()
    fig.savefig(save_path+"/real_sample_MCMC.png", dpi=600)

    #Sample using nested sampling.
    sampler = Sampler(objective)
    fig = sampler.sample()
    fig.savefig(save_path+"/real_sample_nested_sampling.png", dpi=600)

if __name__ == "__main__":
    save_path = "./results/real_data"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data_path = "./data/"
    real_data = ["QCS_03_1uA.dat", "QCS_04_1uA.dat", "QCS_05_1uA.dat",
                 "QCS_06_1uA.dat", "QCS_07_1uA.dat", "QCS_20_1uA.dat"]
    dq  = 2.5
    bkg = 8e-7

    sample_real_data(data_path+"QCS_all.dat", dq, bkg, save_path)
    plot_real_all(data_path+"QCS_all.dat", dq, bkg, save_path)
    plot_real_angles(data_path, real_data, save_path)

    angle_times = {0.3: (93, 1), #Angle: (Points, Time)
                   0.4: (56, 1),
                   0.5: (64, 1),
                   0.6: (70, 1),
                   0.7: (82, 1),
                   2.0: (91, 1)}
    
    plot_simulated_angles(angle_times, dq, bkg, save_path)
