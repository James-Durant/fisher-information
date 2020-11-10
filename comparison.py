import h5py
import numpy as np
import matplotlib.pyplot as plt

from refnx.dataset  import ReflectDataset
from refnx.reflect  import ReflectModel
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
        """Adds background and sample noise to given reflectivity data using the direct beam sample.
    
        Args:
            q (ndarray): the range of q (momentum transfer) values.
            r (ndarray): the range of r (reflectance) values.
            file (string): the file path to the directbeam_noise file.
            constant (int): the sample noise constant.
            bkg_rate (type): the background rate value.
    
        Returns:
            Reflectance values with noise added.
    
        """
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
    sld_bounds   = (0,6.2)
    thick_bounds = (0,220)
    rough_bounds = (1.5,6.5)
    
    def __init__(self, models, datasets):
        self.datasets = [ReflectDataset(data) for data in datasets]
        self.models   = [Fitting.__vary_model(model) for model in models]
    
    @staticmethod
    def __vary_model(model):
        components = model.structure.components
        for i, component in enumerate(components[1:-1]): #Skip over Air/D20 and substrate
            #Set the SLD, thickness and roughness to arbitrary initial values (within their bounds).
            component.sld.real.value = (Fitting.sld_bounds[1]   - Fitting.sld_bounds[0])   / 2
            component.thick.value    = (Fitting.thick_bounds[1] - Fitting.thick_bounds[0]) / 2
            #component.rough.value    = (Fitting.rough_bounds[1] - Fitting.rough_bounds[0]) / 2
            
            component.sld.real.setp(vary=True, bounds=Fitting.sld_bounds)
            component.thick.setp(vary=True, bounds=Fitting.thick_bounds)
            #component.rough.setp(vary=True, bounds=Fitting.rough_bounds)
            
        return model
    
    def __reset_objective(self):
        #Create a list of objectives for each model and use it to create a global objective.
        self.objective = GlobalObjective([Objective(model, data) for (model, data) 
                                          in zip(self.models, self.datasets)])
    def fit_lbfgs(self):
        print("------------------- L-BFGS-B -------------------")
        self.__reset_objective()
        fitter = CurveFitter(self.objective)
        fitter.fit('differential_evolution', polish=False)
        fitter.fit('L-BFGS-B')
        self.display_results()

    def fit_mcmc(self, burn=400, steps=30, nthin=100):
        print("--------------------- MCMC ---------------------")
        self.__reset_objective()
        fitter = CurveFitter(self.objective)
        fitter.sample(burn)
        fitter.reset()
        fitter.sample(steps, nthin=nthin)
        
        #with h5py.File("chains.h5", "w") as file:
        #    file.create_dataset("chain", data=fitter.chain)
        
        self.display_results()
        self.objective.corner()
        plt.show()

    def fit_nested(self):
        print("--------------- Nested Sampling ----------------")
        self.__reset_objective()
        pool = Pool(cpu_count()-1)
        ndim = len(self.objective.varying_parameters())

        sampler = NestedSampler(self.logl, self.objective.prior_transform, ndim, pool=pool, queue_size=cpu_count())
        sampler.run_nested(dlogz=70)
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
    model.fit_lbfgs()
    #model.fit_mcmc()
    #model.fit_nested()
    