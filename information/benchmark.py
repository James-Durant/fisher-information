import numpy as np
import os, time, sys
sys.path.append("../")

from refnx.dataset  import ReflectDataset
from refnx.reflect  import SLD
from refnx.analysis import Objective

from simulation.simulate import simulate_single_contrast
from information.utils   import Sampler, calc_FIM

class Generator:
    """The Generator class contains all code relating random model generation.

    Class Attributes:
        sld_bounds (tuple): the range of values that layer SLDs can take.
        thick_bounds (tuple): the range of values that layer thicknesses can take.
        rough_bounds (tuple): the range of values that layer roughnesses can take.
        substrate_sld (float): the SLD of the substrate (Silicon).
        angle_times (dict): a dictionary of points and times for each measurement angle.

    """
    sld_bounds    = (-1,10)
    thick_bounds  = (20,1000)
    rough_bounds  = (2,8)
    substrate_sld = 2.047
    angle_times   = {0.7: (70, 5),
                     2.0: (70, 20)}

    @staticmethod
    def generate(num_samples, layers):
        """Generates `num_samples` models and data sets with given number of `layers`.

        Args:
            num_samples (int): the number of models to generate.
            layers (int): the number of layers for each model to be generated with.

        Returns:
            models_data (list): `generate_num` models and associated simulated data sets.

        """
        models_data = []
        for i in range(num_samples):
            structure = Generator.random_structure(layers) #Get a random structure.
            #Simulate an experiement using the structure.
            model, data = simulate_single_contrast(structure, Generator.angle_times)
            models_data.append([model, data])
        return models_data

    @staticmethod
    def random_structure(layers):
        """Generates a single random structure with desired number of `layers`.

        Args:
            layers (int): the number of layers for the structure to be generated with.

        Returns:
            (refnx.reflect.Structure): the randomly generated structure.

        """
        #The structure consists of air followed by each layer and then finally the substrate.
        structure = SLD(0, name="Air")
        for i in range(layers):
            layer = Generator.make_component(substrate=False) #Get a random layer.
            #Vary the SLD, thickness and roughness of the layer.
            layer.sld.real.setp(vary=True, bounds=(layer.sld.real.value*0.25, layer.sld.real.value*1.25))
            layer.thick.setp(vary=True, bounds=(layer.thick.value*0.25, layer.thick.value*1.25))
            layer.rough.setp(vary=True, bounds=(layer.rough.value*0.25, layer.rough.value*1.25))
            structure = structure | layer

        return structure | Generator.make_component(substrate=True)

    @staticmethod
    def make_component(substrate):
        """Generates a single layer of a structure.

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

def benchmark(layers, num_samples):
    """Randomly generates `num_samples` samples each of given number of
       `layers` and calculates parameter uncertainness using MCMC sampling,
       nested sampling and the FIM approach for each sample.

    Args:
        layers (int): number of layers for each sample to be generated with.
        num_samples (int): number of samples to generate.

    Returns:
        (tuple): calculation times for MCMC sampling, nested sampling and the FIM approach.

    """
    print("-------- {}-Layer Samples --------".format(layers))

    #Generate the random samples and simulate experiments using them.
    models_data = Generator.generate(num_samples, layers)

    #Iterate over each sample and run MCMC and nested sampling, and calculate the FIM.
    times_MCMC, times_nested, times_FIM = [], [], []
    for i, (model, data) in enumerate(models_data):
        #Create an objective for the random model and simulated data.
        q, r, r_error, counts = data[:,0], data[:,1], data[:,2], data[:,3]
        objective = Objective(model, ReflectDataset([q,r,r_error]))

        #Sample using MCMC and nested sampling.
        sampler = Sampler(objective)

        print("MCMC Sampling {0}/{1}...".format(i+1, len(models_data)))
        _, time_MCMC = sampler.sample_MCMC(fit_first=False)
        times_MCMC.append(time_MCMC)

        print("Nested Sampling {0}/{1}...".format(i+1, len(models_data)))
        _, time_nested = sampler.sample_nested()
        times_nested.append(time_nested)

        #Calculate the FIM.
        print("FIM {0}/{1}...".format(i+1, len(models_data)))
        xi = objective.varying_parameters()
        start = time.time()
        g = calc_FIM(q, xi, counts, model)
        1 / np.diag(g) #Calculate parameter uncertainties.
        end = time.time()
        times_FIM.append(end-start)

    print()
    return times_MCMC, times_nested, times_FIM

if __name__ == "__main__":
    save_path = "./results"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    num_samples = 10
    file = open(save_path+"/benchmark.txt", "w")
    for layer in range(1,7):
        times_MCMC, times_nested, times_FIM = benchmark(layer, num_samples=num_samples)
        file.write("------------------ {}-Layer Samples ------------------\n".format(layer))
        file.write("MCMC Sampling:   {}\n".format(times_MCMC))
        file.write("Nested Sampling: {}\n".format(times_nested))
        file.write("FIM Approach:    {}\n".format(times_FIM))
        file.write("\n")

    file.close()
