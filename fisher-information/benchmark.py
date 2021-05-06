import numpy as np
import os, time, sys
sys.path.append('./') # MCMC sampling cannot find structures without this.

from typing import List, Tuple
from refnx.analysis import Objective

from utils import Sampler, ModelGenerator
from utils import fisher_single_contrast as fisher

def benchmark(layers: int, num_samples: int) -> Tuple[List[float], List[float], List[float]]:
    """Randomly generate `num_samples` samples, each of given number of
       `layers`, and calculate parameter uncertainties using MCMC sampling,
       nested sampling and the FI approach for each sample.

    Args:
        layers (int): number of layers for each sample to be generated with.
        num_samples (int): number of samples to generate.

    Returns:
        times_MCMC (list): calculation times for MCMC sampling.
        times_nested (list): calculation times for nested sampling.
        times_fisher (list): calculation times for FI approach.

    """
    print('-------- {}-Layer Samples --------'.format(layers))

    # Generate random samples and simulate experiments using them.
    generator = ModelGenerator()
    models_data = generator.generate(num_samples, layers)

    # Iterate over each sample, run MCMC and nested sampling, and calculate the FI.
    times_MCMC, times_nested, times_fisher = [], [], []
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

        # Calculate the FI.
        print('>>> FI {0}/{1}'.format(i+1, len(models_data)))
        xi = objective.varying_parameters()
        start = time.time()
        g = fisher(data.x, xi, counts, model)
        np.sqrt(np.diag(g))
        end = time.time()
        times_fisher.append(end-start)

    print()
    return times_MCMC, times_nested, times_fisher

if __name__ == '__main__':
    save_path = './results'
    num_samples = 10

    # Run the benchmark for 10 samples for layers in interval [1,6] (excluding substrate).
    with open(os.path.join(save_path, 'benchmark.txt'), 'w') as file:
        for layer in range(1,7):
            times_MCMC, times_nested, times_fisher = benchmark(layer, num_samples=num_samples)
            file.write('------------------ {}-Layer Samples ------------------\n'.format(layer))
            file.write('MCMC Sampling:   {}\n'.format(times_MCMC))
            file.write('Nested Sampling: {}\n'.format(times_nested))
            file.write('FI Approach:     {}\n'.format(times_fisher))
            file.write('\n')
