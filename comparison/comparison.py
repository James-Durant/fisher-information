import os, sys
sys.path.append("../utils") # Adds higher directory to python modules path.
import numpy as np
import matplotlib.pyplot as plt

from refnx.dataset  import ReflectDataset
from refnx.analysis import Objective, GlobalObjective, CurveFitter

from dynesty import NestedSampler
from dynesty import plotting as dyplot
from dynesty import utils    as dyfunc

from multiprocessing import Pool, cpu_count

from generate import generate_noisy_multiple, vary_model

from structures import thin_layer_sample_1, thin_layer_sample_2
from structures import similar_sld_sample_1, similar_sld_sample_2
from structures import easy_sample_1, many_param_sample, multiple_contrast_sample

class Fitting:
    def __init__(self, save_path, models, datasets):
        self.save_path = save_path
        self.datasets = [ReflectDataset(data) for data in datasets]
        self.models   = models
        for model in models:
           vary_model(model)

    def __reset_objective(self):
        #Create a list of objectives for each model then use this list to create a global objective.
        self.objective = GlobalObjective([Objective(model, data) for (model, data)
                                          in zip(self.models, self.datasets)])
    def fit_lbfgs(self):
        print("\n------------------- L-BFGS-B -------------------")
        self.__reset_objective()
        fitter = CurveFitter(self.objective)
        fitter.fit('differential_evolution', polish=False)
        fitter.fit('L-BFGS-B')
        self.display_results("L-BFGS-B")

    def fit_mcmc(self, burn=400, steps=15, nthin=100):
        print("\n--------------------- MCMC ---------------------")
        self.__reset_objective()
        fitter = CurveFitter(self.objective)
        fitter.fit('differential_evolution')
        fitter.sample(burn)
        fitter.reset()
        fitter.sample(steps, nthin=nthin)

        self.display_results("MCMC")
        self.objective.corner()
        plt.savefig(self.save_path+"/corner_MCMC.png", dpi=300)

    def fit_nested(self):
        print("\n--------------- Nested Sampling ----------------")
        self.__reset_objective()
        pool = Pool(cpu_count()-1)
        ndim = len(self.objective.varying_parameters())

        sampler = NestedSampler(self.logl, self.objective.prior_transform, ndim,
                                pool=pool, queue_size=cpu_count())
        sampler.run_nested()
        pool.close()
        pool.join()

        results = sampler.results
        samples, weights = results.samples, np.exp(results.logwt - results.logz[-1])
        mean, cov = dyfunc.mean_and_cov(samples, weights) #Diagonal elements of covariance matrix, cov, contain parameter uncertainties
        self.logl(mean)

        np.savetxt(self.save_path+"/covar_Nested-Sampling.dat", cov)
        self.plot_objective("Nested-Sampling")
        fig, _ = dyplot.cornerplot(results, color='blue', quantiles=None, show_titles=True, max_n_ticks=3, truths=np.zeros(ndim), truth_color='black')
        plt.savefig(self.save_path+"/corner_Nested-Sampling.png", dpi=300)

    def logl(self, x):
        #Update the model
        for i, parameter in enumerate(self.objective.varying_parameters()):
            parameter.value = x[i]
        return self.objective.logl()

    def display_results(self, fit_method):
        #print(self.objective)
        np.savetxt(self.save_path+"/covar_{}.dat".format(fit_method), self.objective.covar())
        self.plot_objective(fit_method)

    def plot_objective(self, fit_method):
        fig = plt.figure(figsize=[9,7])
        ax = fig.add_subplot(111)

        for objective in self.objective.objectives:
            y, y_err, model = objective._data_transform(model=objective.generative())
            # Add the data in a transformed fashion.
            ax.errorbar(objective.data.x, y, y_err, marker="o", ms=3, lw=0, elinewidth=1, capsize=1.5)
            #Add the prediction/fit
            ax.plot(objective.data.x, model, color="red", zorder=20)

        plt.xlabel("$\mathregular{Q\ (Å^{-1})}$", fontsize=11, weight='bold')
        plt.ylabel('Reflectivity (arb.)',         fontsize=11, weight='bold')
        plt.yscale('log')
        plt.savefig(self.save_path+"/fit_{}.png".format(fit_method), dpi=300)

if __name__ == "__main__":
    """
    Functions for getting structures:
        multiple_contrast_sample
        easy_sample_1
        thin_layer_sample_1
        thin_layer_sample_2
        similar_sld_sample_1
        similar_sld_sample_2
        many_param_sample
    """
    save_path = "./results/easy_sample_1"
    structures = easy_sample_1()
    
    if not os.path.exists(save_path): #Create directory if not present.
        os.makedirs(save_path)
    models, datasets = generate_noisy_multiple(save_path, structures)

    model = Fitting(save_path, models, datasets)
    model.fit_lbfgs()
    model.fit_mcmc(burn=400, steps=15, nthin=100)
    model.fit_nested()
