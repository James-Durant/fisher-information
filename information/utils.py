import numpy as np
import time

from refnx.analysis import CurveFitter
from dynesty import NestedSampler
from dynesty import plotting as dyplot
from dynesty import utils    as dyfunc

def calc_FIM(q, xi, counts, model):
    """Calculates the Fisher information metric (FIM) matrix for a given `model`.

    Args:
        q (numpy.ndarray): array of Q values.
        xi (list): list of refnx Parameter objects representing each varying parameter.
        counts (numpy.ndarray): array of incident neutron counts corresponding to each Q value.
        model (refnx.reflect.ReflectModel): the model to calculate the gradient with.

    Returns:
        numpy.ndarray: FIM matrix for the model and data.

    """
    n = len(q)
    m = len(xi)
    J = np.zeros((n,m))
    #Calculate the gradient of the model reflectivity with every model parameter for every model data point.
    for i in range(n):
        for j in range(m):
            J[i,j] = gradient(model, xi[j], q[i])

    r = model(q) #Use model reflectivity values
    M = np.diag(counts/r, k=0)
    return np.dot(np.dot(J.T, M), J)

def gradient(model, parameter, q_point, step=0.005):
    """Calculate a two-point gradient of `model` reflectivity with model `parameter`.

    Args:
        model (refnx.reflect.ReflectModel): the model to calculate the gradient with.
        parameter (refnx.analysis.Parameter): the parameter to vary when calculating the gradient.
        q_point (float): the Q value of the model reflectivity point to calculate the gradient of.
        step (float): the step size to take when calculating the gradient.

    Returns:
        float: two-point gradient.

    """
    old, step = parameter.value, parameter.value*step #0.5% step by default

    x1 = parameter.value = old - step #First point
    y1 = model(q_point) #Get new r value with altered model.

    x2 = parameter.value = old + step #Second point
    y2 = model(q_point)

    parameter.value = old #Reset parameter
    return (y2-y1) / (x2-x1) #Return the gradient

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

    def sample_MCMC(self, fit_first=True, burn=400, steps=30, nthin=100, verbose=False):
        """Samples the objective using MCMC sampling.

        Args:
            fit_first (Boolean): whether to fit with differential evolution before sampling.
            burn (int): number of samples to use for the burn-in period.
            steps (int): number of steps to use for the main sampling stage.
            nthin (int): amount of thinning to use for the main sampling stage.
            verbose (Boolean): whether to display progress when sampling.

        Returns:
            (matplotlib.pyplot.Figure): MCMC sampling corner plot.
            (float): time taken for sampling.

        """
        start = time.time()
        if fit_first: #Initially fit with differential evolution.
            self.sampler_MCMC.fit('differential_evolution', verbose=verbose)
        self.sampler_MCMC.sample(burn, verbose=verbose) #Burn-in period
        self.sampler_MCMC.reset()
        self.sampler_MCMC.sample(steps, nthin=nthin, verbose=verbose) #Main sampling stage.
        end = time.time() - start
        return self.objective.corner(), end-start

    def sample_nested(self, verbose=False):
        """Samples the objective using nested sampling.

        Args:
            verbose (Boolean): whether to display progress when sampling.

        Returns:
            (matplotlib.pyplot.Figure): nested sampling corner plot.
            (float): time taken for sampling.

        """
        start = time.time()
        self.sampler_nested.run_nested(print_progress=verbose)
        end = time.time()

        results = self.sampler_nested.results
        #Calculate the parameter means.
        weights = np.exp(results.logwt - results.logz[-1])
        mean, _ = dyfunc.mean_and_cov(results.samples, weights)
        self.logl(mean) #Update objective to use mean parameter values.

        fig, _ = dyplot.cornerplot(results, color='blue', quantiles=None, show_titles=True, max_n_ticks=3, truths=np.zeros(self.ndim), truth_color='black')
        return fig, end-start

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
