import numpy as np
import matplotlib.pyplot as plt

from refnx.dataset  import ReflectDataset
from refnx.reflect  import SLD, ReflectModel
from refnx.analysis import Objective, CurveFitter

from dynesty import NestedSampler
from dynesty import plotting as dyplot
from dynesty import utils    as dyfunc
from multiprocessing import Pool, cpu_count

class Data:
    points = 300
    q_min  = 0.005
    q_max  = 0.3
    dq     = 2
    scale  = 1
    bkg    = 1e-7
    
    @staticmethod
    def generate():
        air       = SLD(0)
        layer     = SLD(5)(thick=500, rough=8)
        substrate = SLD(2.047)(thick=0, rough=8)
        structure = air | layer | substrate
        
        model = ReflectModel(structure, scale=Data.scale, dq=Data.dq, bkg=Data.bkg)
    
        q = np.logspace(np.log10(Data.q_min), np.log10(Data.q_max), Data.points)
        r = Data.__add_noise(q, model(q))
        r_error = [1e-10]*Data.points #Change this?
        return [q, r, r_error]
    
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
        for i, r_point in zip(flux_density, r): #Beam interp against simulated reflectance.
            normal_width = r_point * constant / i
            r_noisy.append(np.random.normal(loc=r_point, scale=normal_width)) #Using beam interp
        return r_noisy


class Fitting:
    sld_bounds   = (2,10)
    thick_bounds = (200,700)
    rough_bounds = (6,10)
    
    def __init__(self, data):
        self.data  = ReflectDataset(data)
        self.model = Fitting.__create_model()

    @staticmethod
    def __create_model():
        #Define the model to be fitted to the data here.
        air       = SLD(0, name="Air")
        layer     = SLD(2, name="Layer 1")(thick=250, rough=6)
        substrate = SLD(2.047, name="Substrate")(thick=0, rough=10)
        
        layer.sld.real.setp(bounds=Fitting.sld_bounds, vary=True)
        layer.thick.setp(bounds=Fitting.thick_bounds, vary=True)
        layer.rough.setp(bounds=Fitting.rough_bounds, vary=True)
        substrate.rough.setp(bounds=Fitting.rough_bounds, vary=True)
        
        structure = air | layer | substrate
        return ReflectModel(structure, scale=1, dq=2, bkg=1e-7)
    
    def __reset_objective(self):
        self.objective = Objective(self.model, self.data)
    
    def fit_lbfgs(self):
        print("------------------- L-BFGS-B -------------------")
        self.__reset_objective()
        fitter = CurveFitter(self.objective)
        fitter.fit('differential_evolution', polish=False, verbose=False)
        fitter.fit('L-BFGS-B', verbose=False)
        self.display_results()

    def fit_mcmc(self, burn=500, steps=100, nthin=50):
        print("--------------------- MCMC ---------------------")
        self.__reset_objective()
        fitter = CurveFitter(self.objective)
        fitter.sample(burn)
        fitter.reset()
        fitter.sample(steps, nthin=nthin)
        print(fitter.chain)
        
        self.display_results()
        self.objective.corner()
        plt.show()

    def fit_nested(self):
        print("--------------- Nested Sampling ----------------")
        self.__reset_objective()
        pool = Pool(cpu_count()-1)
        ndim = len(self.objective.varying_parameters())

        sampler = NestedSampler(self.logl, Fitting.prior_transform, ndim, pool=pool, queue_size=cpu_count())
        sampler.run_nested(dlogz=70)
        pool.close()
        pool.join()
        
        results = sampler.results
        samples, weights = results.samples, np.exp(results.logwt - results.logz[-1])
        mean, cov = dyfunc.mean_and_cov(samples, weights) #Diagonal elements of covariance matrix, cov, contain parameter uncertainties
        self.logl(mean)
        print(cov)
        self.plot_objective()
        dyplot.cornerplot(results, color='blue', quantiles=None, show_titles=True, max_n_ticks=3, truths=np.zeros(ndim), truth_color='black')
    
    def logl(self, x):
        #Update the model
        components = self.model.structure.components
        layer = components[1]
        layer.sld.real.value = x[0]
        layer.thick.value    = x[1]
        layer.rough.value    = x[2]
        
        substrate = components[2]
        substrate.rough.value = x[3]
        
        return self.objective.logl()
    
    @staticmethod
    def prior_transform(u):
        x = np.array(u)
        old_range = (0, 1) #Uniform prior
        x[0] = Fitting.convert_range(u[0], old_range, Fitting.sld_bounds)
        x[1] = Fitting.convert_range(u[1], old_range, Fitting.thick_bounds)
        x[2] = Fitting.convert_range(u[2], old_range, Fitting.rough_bounds)
        x[3] = Fitting.convert_range(u[3], old_range, Fitting.rough_bounds)
        return x
    
    @staticmethod
    def convert_range(old_value, old_range, new_range):
        (old_min, old_max) = old_range
        (new_min, new_max) = new_range
        return (((old_value - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min
    
    def display_results(self):
        #components = self.model.structure.components
        #for i, component in enumerate(components[1:]): #Skip over Air
        #    if i != len(components)-2:
        #        print(">>> Layer {0} - SLD:       {1:10.6f} | Error: {2:10.8f}".format(i+1, component.sld.real.value, component.sld.real.stderr))
        #        print(">>> Layer {0} - Thickness: {1:10.6f} | Error: {2:10.8f}".format(i+1, component.thick.value, component.thick.stderr))
        #    print(">>> Layer {0} - Roughness: {1:10.7f} | Error: {2:10.8f}".format(i+1, component.rough.value, component.rough.stderr))
        #print(self.objective)
        print(self.objective.covar(), "\n")
        self.plot_objective()
    
    def plot_objective(self):
        fig = plt.figure(figsize=[9,7])
        ax = fig.add_subplot(111)
        
        y, y_err, model = self.objective._data_transform(model=self.objective.generative())
        # Add the data in a transformed fashion.
        ax.errorbar(self.objective.data.x, y, y_err, label="Data",
                    color="blue", marker="o", ms=3, lw=0, elinewidth=1, capsize=1.5)
        #Add the prediction/fit
        ax.plot(self.objective.data.x, model, color="red", zorder=20, label="Fitted")
        
        plt.xlabel("$\mathregular{Q\ (Å^{-1})}$", fontsize=11, weight='bold')
        plt.ylabel('Reflectivity (arb.)',         fontsize=11, weight='bold')
        plt.yscale('log')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    model = Fitting(Data.generate()) 
    #model.fit_lbfgs()
    #model.fit_mcmc()
    model.fit_nested()
    