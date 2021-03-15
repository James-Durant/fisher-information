import numpy as np

from refnx.analysis import CurveFitter

from dynesty import NestedSampler
from dynesty import plotting as dyplot
from dynesty import utils    as dyfunc

class Sampler:
    def __init__(self, objective):
        self.objective = objective
        self.ndim = len(self.objective.varying_parameters())
        self.sampler_nested = NestedSampler(self.logl, self.objective.prior_transform, self.ndim)
        self.sampler_MCMC = CurveFitter(self.objective)

    def sample_MCMC(self, burn=400, steps=30, nthin=100, verbose=True):
        self.sampler_MCMC.sample(burn, verbose=verbose) #Burn-in period
        self.sampler_MCMC.reset()
        self.sampler_MCMC.sample(steps, nthin=nthin, verbose=verbose) #Main sampling stage.
        return self.objective.corner()

    def sample_nested(self, verbose=True, show_fig=True):
        self.sampler_nested.run_nested(print_progress=verbose)

        results = self.sampler_nested.results
        #Calculate the parameter means.
        weights = np.exp(results.logwt - results.logz[-1])
        mean, _ = dyfunc.mean_and_cov(results.samples, weights)
        self.logl(mean) #Update objective to use mean parameter values.
    
        if show_fig:
            return dyplot.cornerplot(results, color='blue', quantiles=None, 
                                     show_titles=True, max_n_ticks=3, 
                                     truths=np.zeros(self.ndim), truth_color='black')

    def logl(self, x):
        for i, parameter in enumerate(self.objective.varying_parameters()):
            parameter.value = x[i] #Update the model with given parameter values.
        return self.objective.logl()

def calc_FIM(qs, xi, counts, models):
    n = sum(len(q) for q in qs)
    m = len(xi)
    J = np.zeros((n,m))
    
    r_all = []
    start = 0
    for q, model in list(zip(qs, models)):
        for i in range(len(q)):
            for j in range(m):
                J[start+i,j] = gradient(model, xi[j], q[i])
                
        start += len(q)
        r_all.append(model(q))
    
    r = np.concatenate(r_all)
    M = np.diag(counts/r, k=0)
    return np.dot(np.dot(J.T, M), J)

def gradient(model, parameter, q_point, step=0.005):
    old, step = parameter.value, parameter.value*step #0.5% step by default

    x1 = parameter.value = old - step #First point
    y1 = model(q_point) #Get new r value with altered model.

    x2 = parameter.value = old + step #Second point
    y2 = model(q_point)

    parameter.value = old #Reset parameter
    return (y2-y1) / (x2-x1) #Return the gradient

