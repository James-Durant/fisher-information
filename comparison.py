import os
import numpy as np
import matplotlib.pyplot as plt

from refnx.dataset  import ReflectDataset
from refnx.reflect  import SLD, ReflectModel
from refnx.analysis import Objective, GlobalObjective, CurveFitter

from dynesty import NestedSampler
from dynesty import plotting as dyplot
from dynesty import utils    as dyfunc

from multiprocessing import Pool, cpu_count

from structures import multiple_contrast_sample, easy_sample_1, thin_layer_sample_1, thin_layer_sample_2
from structures import similar_sld_sample_1, similar_sld_sample_2, many_param_sample

class Data:
    points = 300
    q_min  = 0.005
    q_max  = 0.3
    dq     = 2
    scale  = 1
    bkg    = 1e-6

    @staticmethod
    def generate(save_path, structures):
        models = []
        datasets = []

        for i, structure in enumerate(structures, 1):
            model = ReflectModel(structure, scale=Data.scale, dq=Data.dq, bkg=Data.bkg)
            models.append(model)

            q = np.logspace(np.log10(Data.q_min), np.log10(Data.q_max), Data.points)
            r, r_error = Data.__add_noise(q, model(q))
            datasets.append([q, r, r_error])
            
            data = np.zeros((Data.points, 3))
            data[:,0] = q
            data[:,1] = r
            data[:,2] = r_error
            np.savetxt(save_path+"/dataset{}.dat".format(i), data, delimiter=",")

        return models, datasets

    @staticmethod    
    def __add_noise(q, r, file="./directbeam_noise.dat", noise_constant=5e5, bkg_rate=5e-7):
        #Try to load the beam sample file: exit the function if not found.
        direct_beam = np.loadtxt(file, delimiter=',')[:, 0:2]
        flux_density = np.interp(q, direct_beam[:, 0], direct_beam[:, 1]) * noise_constant #Not all Q values are the same
    
        #Background signal always ADDs to the signal.
        #Sometimes background could be 0. In which case it does not contribute to the signal
        r = [r_point + max(np.random.normal(1, 0.5) * bkg_rate, 0) for r_point in r]
    
        r_noisy = []
        r_error = []
        for i, r_point in zip(flux_density, r): #Beam interp against simulated reflectance.
            measured_flux = r_point*i
            normal_width = 1 / (noise_constant*np.sqrt(measured_flux))
            r_noisy.append(np.random.normal(loc=r_point, scale=normal_width)) #Using beam interp
            r_error.append(normal_width)
    
        return r_noisy, r_error


class Fitting:
    def __init__(self, save_path, models, datasets):
        self.save_path = save_path
        self.datasets = [ReflectDataset(data) for data in datasets]
        self.models   = models
        for model in models:
            Fitting.__vary_model(model)

    @staticmethod
    def __vary_model(model):
        for component in model.structure.components[1:-1]: #Skip over Air/D20 and substrate
            sld_bounds = (component.sld.real.value*0.75, component.sld.real.value*1.25)
            component.sld.real.setp(vary=True, bounds=sld_bounds)

            thick_bounds = (component.thick.value*0.75, component.thick.value*1.25)
            component.thick.setp(vary=True, bounds=thick_bounds)

            #Set the SLD and thickness to arbitrary initial values (within their bounds).
            component.sld.real.value = sld_bounds[1]
            component.thick.value    = thick_bounds[1]

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
        dyplot.cornerplot(results, color='blue', quantiles=None, show_titles=True, max_n_ticks=3, truths=np.zeros(ndim), truth_color='black')
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
    models, datasets = Data.generate(save_path, structures)

    model = Fitting(save_path, models, datasets)
    model.fit_lbfgs()
    model.fit_mcmc(burn=400, steps=30, nthin=100)
    model.fit_nested()
