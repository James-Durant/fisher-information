import numpy as np
import os, time, sys
sys.path.append('./') # MCMC sampling cannot find structures code without this?

from typing import List, Tuple
from refnx.analysis import Objective

from utils import fisher_single_contrast, Sampler, ModelGenerator

def benchmark(layers: int, num_samples: int
              ) -> Tuple[List[float], List[float], List[float]]:
    """Randomly generate num_samples samples each of given number of
       layers and calculate parameter uncertainness using MCMC sampling,
       nested sampling and the FIM approach for each sample.

    Args:
        layers (int): number of layers for each sample to be generated with.
        num_samples (int): number of samples to generate.

    Returns:
        times_MCMC (list): calculation times for MCMC sampling.
        times_nested (list): calculation times for nested sampling.
        times_FIM (list): calculation times for the FIM approach.

    """
    print('-------- {}-Layer Samples --------'.format(layers))

    # Generate the random samples and simulate experiments using them.
    generator = ModelGenerator()
    models_data = generator.generate(num_samples, layers)

    # Iterate over each sample and run MCMC and nested sampling, and calculate the FIM.
    times_MCMC, times_nested, times_FIM = [], [], []
    for i, (model, data, counts) in enumerate(models_data):
        # Create an objective for the random model and simulated data.
        objective = Objective(model, data)
        sampler = Sampler(objective)

        # Sample using MCMC sampling.
        print('>>> MCMC Sampling {0}/{1}'.format(i+1, len(models_data)))
        start = time.time()
        sampler.sample_MCMC(fit_first=False, show_fig=False, verbose=False)
        end = time.time()
        times_MCMC.append(end-start)

        # Sample using nested sampling.
        print('>>> Nested Sampling {0}/{1}'.format(i+1, len(models_data)))
        start = time.time()
        sampler.sample_nested(show_fig=False, verbose=False)
        end = time.time()
        times_nested.append(end-start)

        # Calculate the FIM.
        print('>>> FIM {0}/{1}'.format(i+1, len(models_data)))
        xi = objective.varying_parameters()
        start = time.time()
        g = fisher_single_contrast(data.x, xi, counts, model)
        1 / np.diag(g) # Calculate parameter uncertainties.
        end = time.time()
        times_FIM.append(end-start)

    print()
    return times_MCMC, times_nested, times_FIM

if __name__ == '__main__':
    save_path = './results'
    num_samples = 10

    # Run the benchmark for 10 samples for layers in the interval [1,6]
    with open(os.path.join(save_path, 'benchmark.txt'), 'w') as file:
        for layer in range(2,7):
            times_MCMC, times_nested, times_FIM = benchmark(layer, num_samples=num_samples)
            file.write('------------------ {}-Layer Samples ------------------\n'.format(layer))
            file.write('MCMC Sampling:   {}\n'.format(times_MCMC))
            file.write('Nested Sampling: {}\n'.format(times_nested))
            file.write('FIM Approach:    {}\n'.format(times_FIM))
            file.write('\n')
