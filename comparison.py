import numpy as np
import matplotlib.pyplot as plt

from refnx.dataset  import ReflectDataset
from refnx.reflect  import SLD, ReflectModel
from refnx.analysis import Objective, GlobalObjective, CurveFitter

from dynesty import NestedSampler
from dynesty import plotting as dyplot
from dynesty import utils    as dyfunc

from multiprocessing import Pool, cpu_count

from structures import multiple_contrast_samples, thin_layer_samples_1, thin_layer_samples_2
from structures import similar_sld_samples_1, similar_sld_samples_2, many_param_samples

class Data:
    points = 300
    q_min  = 0.005
    q_max  = 0.3
    dq     = 2
    scale  = 1
    bkg    = 1e-6
    
    @staticmethod
    def generate(structures):
        models = []
        datasets = []
        
        for structure in structures:
            model = ReflectModel(structure, scale=Data.scale, dq=Data.dq, bkg=Data.bkg)
            models.append(model)
        
            q = np.logspace(np.log10(Data.q_min), np.log10(Data.q_max), Data.points)
            r, r_error = Data.__add_noise(q, model(q))    
            datasets.append([q, r, r_error])

        return models, datasets
    
    @staticmethod
    def __add_noise(q, r, file="./directbeam_noise.dat", constant=100, bkg_rate=5e-7):
        #Try to load the beam sample file: exit the function if not found.
        direct_beam = np.loadtxt(file, delimiter=',')[:, 0:2]
    
        #Background signal always ADDs to the signal.
        #Sometimes background could be 0. In which case it does not contribute to the signal
        r = [r_point + max(np.random.normal(1, 0.5) * bkg_rate, 0) for r_point in r]
        
        flux_density = np.interp(q, direct_beam[:, 0], direct_beam[:, 1]) #Not all Q values are the same
        r_noisy = []
        r_error = []
        for i, r_point in zip(flux_density, r): #Beam interp against simulated reflectance.
            normal_width = r_point * constant / i
            r_noisy.append(np.random.normal(loc=r_point, scale=normal_width)) #Using beam interp
            r_error.append(normal_width)
        return r_noisy, r_error

class Fitting:
    def __init__(self, models, datasets):
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
        print("------------------- L-BFGS-B -------------------")
        self.__reset_objective()
        fitter = CurveFitter(self.objective)
        fitter.fit('differential_evolution', polish=False)
        fitter.fit('L-BFGS-B')
        self.display_results()

    def fit_mcmc(self, burn=1000, steps=30, nthin=100):
        print("--------------------- MCMC ---------------------")
        self.__reset_objective()
        fitter = CurveFitter(self.objective)
        fitter.fit('differential_evolution')
        fitter.sample(burn)
        fitter.reset()
        fitter.sample(steps, nthin=nthin)
    
        self.display_results()
        self.objective.corner()
        plt.show()

    def fit_nested(self):
        print("--------------- Nested Sampling ----------------")
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
        print("Covariance Matrix: \n", cov)
        self.plot_objective()
        dyplot.cornerplot(results, color='blue', quantiles=None, show_titles=True, max_n_ticks=3, truths=np.zeros(ndim), truth_color='black')
    
    def logl(self, x):
        #Update the model
        for i, parameter in enumerate(self.objective.varying_parameters()):
            parameter.value = x[i]
        return self.objective.logl()
    
    def display_results(self):
        #print(self.objective)
        print("Covariance Matrix:\n", self.objective.covar())
        self.plot_objective()
    
    def plot_objective(self):
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
        plt.show()

if __name__ == "__main__":
    """
    Functions for getting structures:
        multiple_contrast_samples
        thin_layer_samples_1
        thin_layer_samples_2
        similar_sld_samples_1
        similar_sld_samples_2
        many_param_samples
    """
    structures = thin_layer_samples_1()
    models, datasets = Data.generate(structures)
    
    model = Fitting(models, datasets) 
    #model.fit_lbfgs()
    model.fit_mcmc()
    #model.fit_nested()
    