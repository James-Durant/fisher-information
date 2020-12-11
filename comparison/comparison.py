import os, sys
sys.path.append("../simulation") # Adds higher directory to python modules path.

import numpy as np
import matplotlib.pyplot as plt

from refnx.dataset  import ReflectDataset
from refnx.analysis import Objective, GlobalObjective, CurveFitter

from multiprocessing import Pool, cpu_count
from dynesty import NestedSampler
from dynesty import plotting as dyplot
from dynesty import utils    as dyfunc

from simulate import simulate_noisy, vary_model

class Fitting:
    """The Fitting class contains the code for fitting a given model and dataset
       with L-BFGS-B, MCMC and nested sampling.

    Attributes:
        save_path (string): path to directory to save figures and results.
        models (list): a list of models to fit.
        datasets (list): a list of datasets ([q,r,r_error]) corresponding to each model.

    """
    def __init__(self, save_path, models, datasets):
        if not os.path.exists(save_path): #Create directory if not present.
            os.makedirs(save_path)

        self.save_path = save_path
        self.datasets = [ReflectDataset(data) for data in datasets]
        self.models   = models
        for model in models: #Vary the SLD and thickness of each layer for each model.
           vary_model(model)

    def __reset_objective(self):
        """Resets the Fitting object's objective so previous results are not re-used."""
        #Create a list of objectives for each model then use this list to create a global objective.
        self.objective = GlobalObjective([Objective(model, data) for (model, data)
                                          in zip(self.models, self.datasets)])
    def fit_lbfgs(self):
        """Fits the given model(s) with the L-BFGS-B algorithm, saves the
           estimated covariance matrix and graphical fit."""
        print("\n------------------- L-BFGS-B -------------------")
        self.__reset_objective() #Clear the objective from any previous fits.
        fitter = CurveFitter(self.objective)
        fitter.fit('differential_evolution', polish=False)
        fitter.fit('L-BFGS-B')
        self.display_results("L-BFGS-B")

    def fit_mcmc(self, burn=400, steps=15, nthin=100):
        """Fits the given model(s) using MCMC sampling, saves the estimated
           covariance matrix, graphical fit and corner plot.

        Args:
            burn (int): number of samples to use for the burn-in period.
            steps (int): number of steps to use for the main sampling stage.
            nthin (int): amount of thinning to use for the main sampling stage.

        """
        print("\n--------------------- MCMC ---------------------")
        self.__reset_objective() #Clear the objective from any previous fits.
        fitter = CurveFitter(self.objective)
        #Initially fit using differential evolution and then sample around the result.
        fitter.fit('differential_evolution')
        fitter.sample(burn) #Burn-in period
        fitter.reset()
        fitter.sample(steps, nthin=nthin) #Main sampling stage.

        self.display_results("MCMC")
        self.objective.corner() #Use corner.py to generate the corner plot.
        plt.savefig(self.save_path+"/corner_MCMC.png", dpi=600)

    def fit_nested(self):
        """Fits the given model(s) using nested sampling, saves the estimated
           covariance matrix, graphical fit and corner plot."""
        print("\n--------------- Nested Sampling ----------------")
        self.__reset_objective() #Clear the objective from any previous fits.
        pool = Pool(cpu_count()-1) #Use multi-processing with pools.
        ndim = len(self.objective.varying_parameters())

        sampler = NestedSampler(self.logl, self.objective.prior_transform, ndim,
                                pool=pool, queue_size=cpu_count())
        sampler.run_nested()
        pool.close()
        pool.join()

        results = sampler.results
        #Calculate the parameter means and covariance matrix.
        samples, weights = results.samples, np.exp(results.logwt - results.logz[-1])
        mean, cov = dyfunc.mean_and_cov(samples, weights)
        self.logl(mean) #Set the objective to use these mean values.

        np.savetxt(self.save_path+"/covar_Nested-Sampling.dat", cov)
        self.plot_objective("Nested-Sampling")
        fig, _ = dyplot.cornerplot(results, color='blue', quantiles=None, show_titles=True, max_n_ticks=3, truths=np.zeros(ndim), truth_color='black')
        plt.savefig(self.save_path+"/corner_Nested-Sampling.png", dpi=600)

    def logl(self, x):
        """Calculates the log-likelihood of the parameters `x` against the model(s)
           and dataset(s)

        Args:
            x (numpy.ndarray): array of parameter values.

        Returns:
            float: log-likelihood of the parameters `x`.

        """
        for i, parameter in enumerate(self.objective.varying_parameters()):
            parameter.value = x[i] #Update the model with given parameter values.
        return self.objective.logl()

    def display_results(self, fit_method):
        """Saves the covariance matrix estimate after fitting and plots the fit.

        Args:
            fit_method (string): the method used to fit the data.

        """
        np.savetxt(self.save_path+"/covar_{}.dat".format(fit_method), self.objective.covar())
        self.plot_objective(fit_method)

    def plot_objective(self, fit_method):
        """Graphically plots the fit(s) against the dataset(s).

        Args:
            fit_method (string): the method used to fit the data.

        """
        fig = plt.figure(figsize=[9,7])
        ax = fig.add_subplot(111)

        #Plot each objective fit against the corresponding dataset.
        for objective in self.objective.objectives:
            # Add the data in a transformed fashion.
            y, y_err, model = objective._data_transform(model=objective.generative())
            ax.errorbar(objective.data.x, y, y_err, marker="o", ms=3, lw=0, elinewidth=1, capsize=1.5)
            #Add the fit
            ax.plot(objective.data.x, model, color="red", zorder=20)

        plt.xlabel("$\mathregular{Q\ (Å^{-1})}$", fontsize=11, weight='bold')
        plt.xlim(0, 0.25)
        plt.ylabel('Reflectivity (arb.)',         fontsize=11, weight='bold')
        plt.yscale('log')
        plt.savefig(self.save_path+"/fit_{}.png".format(fit_method), dpi=600)


def run_experiment(structures, points, angle_times, save_path):
    """Simulates an experiment for a given structure over a number of angles,
       each having its own measurement time.

    Args:
        structures (list): list of refnx.reflect.struture objects corresponding to each contrast.
        points (int): number of points to simulate per measurement angle.
        angle_times (dict): dictionary measurement times per angle.
        save_path (string): path to directory to save each contrast's data to.

    Returns:
       models (list): list of refnx.reflect.ReflectModel objects, one for each contrast.
       datasets (list): list of [q,r,r_error] lists for each contrast.

    """
    if not os.path.exists(save_path): #Create directory if not present.
        os.makedirs(save_path)

    models, datasets = [], []
    #For each contrast, simulate an experiment using angles with different measurement times.
    for i, structure in enumerate(structures, 1):
        q, r, r_error = [], [], []
        for angle in angle_times:
            #Simulate the experiment
            time = angle_times[angle]
            model, q_anlge, r_angle, r_error_angle, _ = simulate_noisy(structure, angle, points, time)

            #Combine the q, r and r_error values with the data of other angles.
            q += q_anlge
            r += r_angle
            r_error += r_error_angle

        #Save combined dataset to .dat file
        data = np.zeros((len(angle_times)*points, 3))
        data[:,0] = q
        data[:,1] = r
        data[:,2] = r_error
        data = data[data[:,0].argsort()] #Sort by Q
        np.savetxt(save_path+"/contrast_{}.dat".format(i), data, delimiter=",")

        models.append(model)
        datasets.append([data[:,0], data[:,1], data[:,2]])

    return models, datasets

if __name__ == "__main__":
    from structures import similar_sld_sample_1, similar_sld_sample_2
    from structures import thin_layer_sample_1,  thin_layer_sample_2
    from structures import easy_sample, many_param_sample, multiple_contrast_sample

    structure   = easy_sample #Choose structure here.
    points      = 75          #Number of points per angle.
    angle_times = {0.3: 5, 0.7: 10, 2.0: 40} #Measurement time for each angle.
    save_path   = "./results/"+structure.__name__

    models, datasets = run_experiment(structure(), points, angle_times, save_path)

    #Fit the simulated data using L-BFGS-B, MCMC and/or nested sampling.
    fitter = Fitting(save_path, models, datasets)
    fitter.fit_lbfgs()
    fitter.fit_mcmc(burn=400, steps=15, nthin=100)
    fitter.fit_nested()
