import numpy as np
import os

from typing import Callable
from refnx.analysis import Objective, CurveFitter

from simulate import AngleTimes
from simulate import simulate_single_contrast as simulate

from utils import vary_structure
from utils import fisher_single_contrast as fisher

def compare_fit_variance(structure: Callable, angle_times: AngleTimes,
                         save_path: str, n: int=500) -> None:
    """Compares the inverse FIM with the variance in parameter estimation
       using established fitting algorithms for `n` fits.

    Args:
        structure (function): structure to simulate experiment on.
        angle_times (dict): points and simulation times for each angle.
        save_path (str): path to directory to save FIM and fit variances.
        n (int): number of fits to run.

    """
    # Fit `n` times
    param_estimates, inv_FIM = [], []
    for i in range(n):
        # Simulate the experiment using the given angles, number of points and times.
        model, data, counts = simulate(structure(), angle_times, include_counts=True)

        # Vary the layers' SLDs and thicknesses and set them to random values.
        vary_structure(model.structure, random_init=True)

        # Fit the model using differential evolution.
        objective = Objective(model, data)
        CurveFitter(objective).fit('differential_evolution', verbose=False)

        # Get the estimated parameter values.
        xi = objective.varying_parameters()
        param_estimates.append([param.value for param in xi])

        # Calculate the FIM matrix and FIM parameter variances.
        g = fisher(data.x, xi, counts, model)
        inv_FIM.append(1 / np.diag(g))

        # Display progress every 10 fits.
        if i % 10 == 0:
            print('>>> {0}/{1}'.format(i, n))

    # Calculate the variances in parameter estimates from the `n` fits.
    param_vars = np.var(np.array(param_estimates), axis=0)
    # Calculate the mean inverse FIM for each parameter.
    mean_inv_FIM = np.mean(np.array(inv_FIM), axis=0)

    # Create directory to save to if not present.
    file_path = os.path.join(save_path, structure.__name__)
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    # Save the results to .txt file.
    with open(os.path.join(file_path, 'variances.txt'), 'w') as file:
        file.write('Variance in Parameter Estimation:\n')
        file.write(str(param_vars)+'\n'*2)
        file.write('Mean Inverse FIM:\n')
        file.write(str(mean_inv_FIM))

if __name__ == '__main__':
    from structures import similar_sld_sample_1, similar_sld_sample_2
    from structures import thin_layer_sample_1, thin_layer_sample_2
    from structures import easy_sample, QCS_sample, many_param_sample

    save_path = './results'

    structure = easy_sample # Choose structure here.
    angle_times = {0.7: (70, 5), # Angle: (Points, Time)
                   2.0: (70, 20)}

    compare_fit_variance(structure, angle_times, save_path, n=500)
