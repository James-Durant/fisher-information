import numpy as np
import os

from refnx.analysis import Objective, CurveFitter
from simulate import simulate

from utils import Sampler, vary_structure

def fitting_sampling_biases(structure, angle_times, n):
    true = []
    for component in structure()[1:-1]:
        true.append(component.thick.value)
        true.append(component.sld.real.value)
    true = np.asarray(true)

    fitted_params = []
    sampled_params = []
    for i in range(n):
        objective = Objective(*simulate(structure(), angle_times))
        print("Ground Truth - Chi-squared: {0}, Log-likelihood: {1}".format(round(objective.chisqr(), 4), round(objective.logl(), 4)))
        
        vary_structure(objective.model.structure, random=True, bound_size=0.25)
        
        CurveFitter(objective).fit('differential_evolution', verbose=False)
        print("Fitted - Chi-squared: {0}, Log-likelihood: {1}".format(round(objective.chisqr(), 4), round(objective.logl(), 4)))
        fitted_params.append([param.value for param in objective.varying_parameters()])

        Sampler(objective).sample_nested(verbose=False, show_fig=False)
        print("Sampled - Chi-squared: {0}, Log-likelihood: {1}".format(round(objective.chisqr(), 4), round(objective.logl(), 4)))
        sampled_params.append([param.value for param in objective.varying_parameters()])
        
        print("{0}/{1}...\n".format(i+1, n))

    fitting_biases = np.mean(fitted_params, axis=0) - true
    sampling_biases = np.mean(sampled_params, axis=0) - true
    names = [param.name for param in objective.varying_parameters()]

    save_path = "./results/fitting_sampling_biases"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    with open(save_path+"/{}.txt".format(structure.__name__), "w") as file:
        file.write("---------- Fitting Biases ----------\n")
        for name, bias in list(zip(names, fitting_biases)):
            file.write("{0}: {1}\n".format(name, bias))
    
        file.write("\n---------- Sampling Biases ----------\n")   
        for name, bias in list(zip(names, sampling_biases)):
            file.write("{0}: {1}\n".format(name, bias))

if __name__ == "__main__":
    from structures import similar_sld_sample_1, similar_sld_sample_2
    from structures import thin_layer_sample_1, thin_layer_sample_2
    from structures import easy_sample, many_param_sample

    structure   = easy_sample
    angle_times = {0.7: (70, 5), #Angle: (Points, Time)
                   2.0: (70, 20)}

    fitting_sampling_biases(structure, angle_times, 100)
