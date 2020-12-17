import os, sys
sys.path.append("../simulation") #Adds directory to Python modules path.

import numpy as np
import matplotlib.pyplot as plt

from refnx.dataset  import ReflectDataset
from refnx.analysis import Objective, GlobalObjective, CurveFitter

from multiprocessing import Pool, cpu_count
from dynesty import NestedSampler
from dynesty import plotting as dyplot
from dynesty import utils    as dyfunc

from simulate import simulate_single_contrast, simulate_multiple_contrasts, vary_model

class Fitting:
    """The Fitting class contains the code for fitting a given model and dataset
       with differential evolution, MCMC and nested sampling.

    Attributes:
        save_path (string): path to directory to save figures and results.
        models (list): a list of models to fit.
        datasets (list): a list of datasets corresponding to each model.

    """
    def __init__(self, save_path, models, datasets):
        self.save_path = save_path
        self.datasets = [ReflectDataset([data[:,0], data[:,1], data[:,2]]) for data in datasets]
        self.models   = models
        for model in models: #Vary the SLD and thickness of each layer for each model.
           vary_model(model)

    def __reset_objective(self):
        """Resets the Fitting object's objective so previous results are not re-used."""
        #Create a list of objectives for each model then use this list to create a global objective.
        self.objective = GlobalObjective([Objective(model, data) for (model, data)
                                          in zip(self.models, self.datasets)])
    def fit_evolution(self):
        """Fits the given model(s) with the differential evolution algorithm,
           saves the estimated covariance matrix and graphical fit."""
        print("\n------------ Differential Evolution ------------")
        self.__reset_objective() #Clear the objective from any previous fits.
        fitter = CurveFitter(self.objective)
        fitter.fit('differential_evolution')
        self.display_results("differential_evolution")

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
        fig = self.objective.corner() #Use corner.py to generate the corner plot.
        fig.savefig(self.save_path+"/corner_MCMC.png", dpi=600)

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

        np.savetxt(self.save_path+"/covar_nested_sampling.dat", cov)
        self.plot_objective("nested_sampling")
        fig, _ = dyplot.cornerplot(results, color='blue', quantiles=None, show_titles=True, max_n_ticks=3, truths=np.zeros(ndim), truth_color='black')
        fig.savefig(self.save_path+"/corner_nested_sampling.png", dpi=600)

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
        fig = plt.figure(figsize=[9,7], dpi=600)
        ax = fig.add_subplot(111)

        #Plot each objective fit against the corresponding dataset.
        for objective in self.objective.objectives:
            # Add the data in a transformed fashion.
            y, y_err, model = objective._data_transform(model=objective.generative())
            ax.errorbar(objective.data.x, y, y_err, marker="o", ms=3, lw=0, elinewidth=1, capsize=1.5)
            #Add the fit
            ax.plot(objective.data.x, model, color="red", zorder=20)

        ax.set_xlabel("$\mathregular{Q\ (Å^{-1})}$", fontsize=11, weight='bold')
        ax.set_xlim(0, 0.3)
        ax.set_ylabel('Reflectivity (arb.)',         fontsize=11, weight='bold')
        ax.set_yscale('log')
        fig.savefig(self.save_path+"/fit_{}.png".format(fit_method), dpi=600)

if __name__ == "__main__":
    from structures import similar_sld_sample_1, similar_sld_sample_2
    from structures import thin_layer_sample_1,  thin_layer_sample_2
    from structures import easy_sample, many_param_sample, multiple_contrast_sample

    structure   = easy_sample #Choose structure here.
    angle_times = {0.7: (70, 5), #Angle: (Points, Time)
                   2.0: (70, 20)}

    save_path = "./results/"+structure.__name__
    if not os.path.exists(save_path): #Create directory if not present.
        os.makedirs(save_path)

    if structure == multiple_contrast_sample:
        models, datasets = simulate_multiple_contrasts(structure(), angle_times, save_path=save_path)
    else:
        model, data = simulate_single_contrast(*structure(), angle_times, save_path=save_path)
        models, datasets = [model], [data]

    #Fit the simulated data using differential evolution, MCMC and/or nested sampling.
    fitter = Fitting(save_path, models, datasets)
    fitter.fit_evolution()
    fitter.fit_mcmc(burn=400, steps=15, nthin=100)
    fitter.fit_nested()
