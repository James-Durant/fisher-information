import os, sys, time
sys.path.append("../simulation") #Adds directory to Python modules path.

import numpy as np

from refnx.dataset  import ReflectDataset
from refnx.reflect  import SLD
from refnx.analysis import Objective, CurveFitter

from dynesty import NestedSampler

from simulate import simulate_single_contrast
from fisher   import calc_FIM

class Sampler:
    """The Sampler class contains the code for MCMC and nested sampling.

    Attributes:
        objective (refnx.analysis.Objective): the objective to fit.
        ndim (int): the dimensionality of the problem.
        sampler_nested (dynesty.NestedSampler): dynesty static nested sampler.
        sampler_MCMC (refnx.analysis.CurveFitter): refnx curve fitter for MCMC sampling.

    """
    def __init__(self, objective):
        self.objective = objective
        self.ndim = len(objective.varying_parameters())
        self.sampler_nested = NestedSampler(self.logl, self.objective.prior_transform, self.ndim)
        self.sampler_MCMC   = CurveFitter(self.objective)

    def sample_MCMC(self, burn=400, steps=15, nthin=100):
        """Samples the objective using MCMC sampling.

        Args:
            burn (int): number of samples to use for the burn-in period.
            steps (int): number of steps to use for the main sampling stage.
            nthin (int): amount of thinning to use for the main sampling stage.

        Returns:
            (float): time taken for the MCMC sampling.

        """
        start = time.time()
        self.sampler_MCMC.sample(burn, verbose=False) #Burn-in period
        self.sampler_MCMC.reset()
        self.sampler_MCMC.sample(steps, nthin=nthin, verbose=False) #Main sampling stage.
        return time.time() - start

    def sample_nested(self):
        """Samples the objective using nested sampling.

        Returns:
            (float): time taken for the nested sampling.

        """
        start = time.time()
        self.sampler_nested.run_nested(print_progress=False)
        return time.time() - start

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

class Generator:
    """The Generator class contains all code relating random model generation.

    Class Attributes:
        sld_bounds (tuple): the range of values layer SLDs can take.
        thick_bounds (tuple): the range of values layer thicknesses can take.
        rough_bounds (tuple): bounds on the roughness between each layer.
        substrate_sld (float): the SLD of the substrate.
        dq (int): instrument resolution parameter.
        bkg (float): background to apply when generating.
        bkg_rate (float): background rate used for adding background noise.

    """
    sld_bounds    = (-1,10)
    thick_bounds  = (20,1000)
    rough_bounds  = (2,8)
    substrate_sld = 2.047
    dq            = 2
    bkg           = 1e-7
    bkg_rate      = 1e-6
    angle_times   = {0.7: (70, 5),
                     2.0: (70, 20)}

    @staticmethod
    def generate(num_samples, layers):
        """Generates `num_samples` models and data sets with given number of layers.

        Args:
            num_samples (int): the number of models to generate.
            layers (int): the number of layers for each model to be generated with.

        Returns:
            models (list): `generate_num` models and associated simulated data sets.

        """
        models = []
        for i in range(num_samples):
            structure = Generator.__random_structure(layers) #Get a random structure.
            #Simulate an experiement using the structure.
            model, data = simulate_single_contrast(structure, Generator.angle_times,
                                                    dq=Generator.dq, bkg=Generator.bkg,
                                                    bkg_rate=Generator.bkg_rate)
            models.append([model, data])
        return models

    @staticmethod
    def __random_structure(layers):
        """Generates a single random structure with desired number of `layers`.

        Args:
            layers (int): the number of layers for each structure to be generated with.

        Returns:
            (refnx.reflect.Structure): the randomly generated structure.

        """
        #The structure consists of air followed by each layer and then finally the substrate.
        structure = SLD(0, name="Air")
        for i in range(layers):
            layer = Generator.__make_component(substrate=False)
            #Vary the SLD, thickness and roughness of the layer.
            layer.sld.real.setp(vary=True, bounds=(layer.sld.real.value*0.25, layer.sld.real.value*1.25))
            layer.thick.setp(vary=True, bounds=(layer.thick.value*0.25, layer.thick.value*1.25))
            layer.rough.setp(vary=True, bounds=(layer.rough.value*0.25, layer.rough.value*1.25))

            structure = structure | layer

        return structure | Generator.__make_component(substrate=True)

    @staticmethod
    def __make_component(substrate=False):
        """Generates a single a layer of a structure.

        Args:
            substrate (Boolean): whether the component is the substrate or not.

        Returns:
            (refnx.reflect.Component): the randomly generated layer.

        """
        if substrate:
            thickness = 0 #Substrate has 0 thickness in refnx.
            sld       = Generator.substrate_sld
        else:
            #Select a random thickness and SLD.
            thickness = np.random.choice(np.arange(*Generator.thick_bounds, 5))
            sld       = np.random.choice(np.arange(*Generator.sld_bounds, 0.1))

        #Select a random roughness for the layer.
        roughness = np.random.choice(np.arange(*Generator.rough_bounds, 0.5))
        return SLD(sld)(thick=thickness, rough=roughness)

def benchmark(layers, num_samples=100):
    """Randomly generates `num_samples` samples each of given number of
       `layers` and calculates parameter uncertatines using MCMC sampling,
       nested sampling and the FIM approach.

    Args:
        layers (int): number of layers for each sample to be generated with.
        num_samples (int): number of samples to generate.

    Returns:
        (tuple): mean calculation time for MCMC sampling, nested sampling and the FIM.

    """
    print("-------- {}-Layer Samples --------".format(layers))
    models = Generator.generate(num_samples, layers)
    times_MCMC, times_nested, times_FIM = [], [], []
    n = len(models)
    for i, (model, data) in enumerate(models):
        q, r, r_error, flux = data[:,0], data[:,1], data[:,2], data[:,3]
        objective = Objective(model, ReflectDataset([q,r,r_error]))

        sampler = Sampler(objective)
        print("MCMC Sampling {0}/{1}...".format(i+1, n))
        times_MCMC.append(sampler.sample_MCMC())
        print("Nested Sampling {0}/{1}...".format(i+1, n))
        times_nested.append(sampler.sample_nested())

        print("FIM {0}/{1}...".format(i+1, n))
        xi = objective.varying_parameters()
        start = time.time()
        g = calc_FIM(q, xi, flux, model)
        1 / np.diag(g) #Calculate parameter uncertainties.
        end = time.time()
        times_FIM.append(end-start)
        
    print()
    return np.mean(times_MCMC), np.mean(times_nested), np.mean(times_FIM)

if __name__ == "__main__":
    save_path = "./results"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    file = open(save_path+"/benchmark.txt", "w")
    
    num_samples = 1
    for layer in range(1,2):
        time_MCMC, time_nested, time_FIM = benchmark(layer, num_samples=num_samples)
        file.write("------------------ {}-Layer Samples ------------------\n".format(layer))
        file.write("MCMC Sampling:   {}s\n".format(time_MCMC))
        file.write("Nested Sampling: {}s\n".format(time_nested))
        file.write("FIM Approach:    {}s\n".format(time_FIM))
        file.write("\n")
        
    file.close()
