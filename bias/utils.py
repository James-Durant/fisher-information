import numpy as np

from refnx.reflect import ReflectModel
from refnx.analysis import CurveFitter

from dynesty import NestedSampler, DynamicNestedSampler
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc

class Sampler:
    def __init__(self, objective):
        self.objective = objective
        self.sampler_MCMC = CurveFitter(self.objective)
        
        self.ndim = len(self.objective.varying_parameters())
        self.sampler_nested_static = NestedSampler(self.logl, self.objective.prior_transform, self.ndim)
        self.sampler_nested_dynamic = DynamicNestedSampler(self.logl, self.objective.prior_transform, self.ndim)

    def sample_MCMC(self, burn=400, steps=30, nthin=100, 
                    fit_first=True, verbose=True, show_fig=True):
        if fit_first:
           self.sampler_MCMC.fit('differential_evolution', verbose=verbose)
           
        self.sampler_MCMC.sample(burn, verbose=verbose)
        self.sampler_MCMC.reset()
        self.sampler_MCMC.sample(steps, nthin=nthin, verbose=verbose)
        
        if show_fig:
            return self.objective.corner()

    def sample_nested(self, dynamic=False, verbose=True, show_fig=True):
        if dynamic:
            self.sampler_nested_dynamic.run_nested(print_progress=verbose, wt_kwargs={'pfrac': 1.0})
        else:
            self.sampler_nested_static.run_nested(print_progress=verbose)

        results = self.sampler_nested.results
        weights = np.exp(results.logwt - results.logz[-1])
        mean, _ = dyfunc.mean_and_cov(results.samples, weights)
        self.logl(mean)
    
        if show_fig:
            return dyplot.cornerplot(results, color='blue', quantiles=None, 
                                     show_titles=True, max_n_ticks=3, 
                                     truths=np.zeros(self.ndim), 
                                     truth_color='black')[0]

    def logl(self, x):
        for i, parameter in enumerate(self.objective.varying_parameters()):
            parameter.value = x[i]
        return self.objective.logl()

def calc_FIM(qs, xi, counts, models):
    if not any(isinstance(x, np.ndarray) for x in qs):
        qs = [qs]
    
    if not any(isinstance(x, np.ndarray) for x in counts):
        counts = np.asarray(counts)
    else:
        counts = np.concatenate(counts)
    
    if isinstance(models, ReflectModel):
        models = [models]
    
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
    M = np.diag(counts / r, k=0)
    return np.dot(np.dot(J.T, M), J)

def gradient(model, parameter, q_point, step=0.005):
    old, step = parameter.value, parameter.value*step

    x1 = parameter.value = old - step
    y1 = model(q_point)

    x2 = parameter.value = old + step
    y2 = model(q_point)

    parameter.value = old
    return (y2-y1) / (x2-x1)

def vary_structure(structure, random_init=False, bound_size=0.2):
    for component in structure.components[1:-1]:
        sld_bounds = (component.sld.real.value*(1-bound_size), component.sld.real.value*(1+bound_size))
        thick_bounds = (component.thick.value*(1-bound_size), component.thick.value*(1+bound_size))

        component.sld.real.setp(vary=True, bounds=sld_bounds)
        component.thick.setp(vary=True, bounds=thick_bounds)
        
        if random_init:
            component.sld.real.value = np.random.uniform(*sld_bounds)
            component.thick.value = np.random.uniform(*thick_bounds)

    return structure