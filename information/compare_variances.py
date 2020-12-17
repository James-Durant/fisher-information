import os, sys
sys.path.append("../simulation") #Adds directory to Python modules path.

import numpy as np

from refnx.dataset  import ReflectDataset
from refnx.analysis import Objective, CurveFitter

from fisher   import calc_FIM
from simulate import simulate_single_contrast, vary_model

def compare_fit_variance(structure, angle_times, save_path, n=1000):
    """Compares the multiplicative inverse Fisher information (variance) with
       variance in parameter estimation using traditional fitting for `n` fits.

    Args:
        structure (refnx.reflect.Structure): the structure to simulate the experiment for.
        angle_times (dict): dictionary of number of points and measurement times to use for each angle.
        save_path (string): path to directory to save Fisher and fit variances.
        n (int): number of fits to run.

    """
    param_estimates, inv_fisher_infos = [], []
    for i in range(n): #Fit `n` times.
        if i % 10 == 0: #Display progress every 10 fits.
            print("{0}/{1}...".format(i, n))

        #Simulate the experiment using the given angle, number of points and time.
        model, data = simulate_single_contrast(*structure(), angle_times)
        vary_model(model) #Vary the SLD and thickness of each layer and set them to random values.

        q, r, r_error, flux = data[:,0], data[:,1], data[:,2], data[:,3]

        #Fit the model using differential evolution.
        objective = Objective(model, ReflectDataset([q, r, r_error]))
        fitter = CurveFitter(objective)
        fitter.fit('differential_evolution', verbose=False)

        xi = objective.varying_parameters() #Get the parameter estimates.
        param_estimates.append([param.value for param in xi])
        
        g = calc_FIM(q, xi, flux, model) #Calculate the Fisher information matrix.
        inv_fisher_infos.append(1 / np.diag(g))

    #Calculate the variances in parameter estimates from `n` fits.
    param_vars = np.var(np.array(param_estimates), axis=0)
    #Calculate the mean inverse Fisher information for each parameter.
    inv_fisher = np.mean(np.array(inv_fisher_infos), axis=0)
    print("Variance in Parameter Estimation:", param_vars)
    print("Inverse Fisher Information:", inv_fisher)

    #Save the results to a .txt file.
    with open(save_path+"/variance_comparison.txt", "w") as file:
        file.write("Variance in Parameter Estimation:\n")
        file.write(str(param_vars)+"\n"*2)
        file.write("Inverse Fisher Information:\n")
        file.write(str(inv_fisher))

if __name__ == "__main__":
    from structures import similar_sld_sample_1, similar_sld_sample_2
    from structures import thin_layer_sample_1,  thin_layer_sample_2
    from structures import easy_sample, many_param_sample

    structure   = easy_sample #Choose structure here.
    angle_times = {0.7: (70, 5), #Angle: (Points, Time)
                   2.0: (70, 20)}

    save_path = "./results/"+structure.__name__
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    compare_fit_variance(structure, angle_times, save_path, n=500)