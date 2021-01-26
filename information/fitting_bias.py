import numpy as np
import os, sys
sys.path.append("../")

from refnx.dataset  import ReflectDataset
from refnx.analysis import Objective, CurveFitter

from simulation.simulate import simulate_single_contrast, vary_model

def fiting_bias(structure, angle_times, save_path, n):
    """Calculates the bias in parameter estimation using traditional fitting for `n` fits.

    Args:
        structure (function): the structure to simulate the experiment on.
        angle_times (dict): dictionary of points and measurement times to use for each angle.
        save_path (string): path to directory to save fitting biases to.
        n (int): number of fits to run.

    """
    #Get the thickness and SLDs of the ground truth model.
    true = []
    for component in structure()[1:-1]:
        true.append(component.thick.value)
        true.append(component.sld.real.value)

    fitted_params = []
    for i in range(n): #Fit `n` times
        if i % 10 == 0: #Display progress every 10 fits.
            print("{0}/{1}...".format(i, n))

        #Simulate the experiment using the given angle, number of points and time.
        model, data = simulate_single_contrast(structure(), angle_times)
        vary_model(model) #Vary the SLD and thickness of each layer and set them to random values.

        q, r, r_error = data[:,0], data[:,1], data[:,2]
        objective = Objective(model, ReflectDataset([q,r,r_error]))

        #Fit the model using differential evolution.
        fitter = CurveFitter(objective)
        fitter.fit('differential_evolution', verbose=False)
        #Recored the fitted parameters.
        fitted_params.append([param.value for param in objective.varying_parameters()])

    #Calculate the bias in each fitted parameter.
    biases = np.mean(fitted_params, axis=0) - np.array(true)
    names  = [param.name for param in objective.varying_parameters()]

    #Save the biases to a .txt file.
    with open(save_path+"/fitting_biases.txt", "w") as file:
        file.write("---------- Fitting Biases ----------\n")
        for name, bias in list(zip(names, biases)):
            file.write("{0}: {1}\n".format(name, bias))

if __name__ == "__main__":
    from simulation.structures import similar_sld_sample_1, similar_sld_sample_2
    from simulation.structures import thin_layer_sample_1,  thin_layer_sample_2
    from simulation.structures import easy_sample, many_param_sample

    structure   = easy_sample #Choose structure here.
    angle_times = {0.7: (70, 5), #Angle: (Points, Time)
                   2.0: (70, 20)}

    save_path = "./results/"+structure.__name__
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    fits = 1000
    fiting_bias(structure, angle_times, save_path, fits)
