import numpy as np
from refnx.analysis import Objective
from simulate import simulate

from utils import Sampler

def vary_structure(structure):
    for component in structure[1:-1]:
        sld_bounds = (component.sld.real.value*0.9, component.sld.real.value*1.1)
        thick_bounds = (component.thick.value*0.9, component.thick.value*1.1)

        component.sld.real.setp(vary=True, bounds=sld_bounds)
        component.thick.setp(vary=True, bounds=thick_bounds)

        component.sld.real.value = np.random.uniform(*sld_bounds)
        component.thick.value = np.random.uniform(*thick_bounds)
    
    return structure

def sampling_bias(structure, angle_times, n):
    #Get the thickness and SLDs of the ground truth model.
    true = []
    for component in structure()[1:-1]:
        true.append(component.thick.value)
        true.append(component.sld.real.value)

    fitted_params = []
    for i in range(n):
        objective = Objective(*simulate(vary_structure(structure()), angle_times))

        sampler = Sampler(objective)
        sampler.sample_nested(verbose=False, show_fig=False)
        fitted_params.append([param.value for param in objective.varying_parameters()])
        
        print("{0}/{1}...".format(i+1, n))

    #Calculate the bias in each fitted parameter.
    biases = np.mean(fitted_params, axis=0) - np.asarray(true)
    names = [param.name for param in objective.varying_parameters()]

    with open("sampling_biases_{}.txt".format(structure.__name__), "w") as file:
        file.write("---------- Sampling Biases ----------\n")
        for name, bias in list(zip(names, biases)):
            file.write("{0}: {1}\n".format(name, bias))

if __name__ == "__main__":
    from structures import similar_sld_sample_1, similar_sld_sample_2
    from structures import thin_layer_sample_1, thin_layer_sample_2
    from structures import easy_sample, many_param_sample

    structure   = thin_layer_sample_1
    angle_times = {0.7: (70, 5), #Angle: (Points, Time)
                   2.0: (70, 20)}

    sampling_bias(structure, angle_times, 100)
