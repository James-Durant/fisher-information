import sys
sys.path.append("../simulation") # Adds higher directory to python modules path.

import numpy as np
import matplotlib.pyplot as plt

from refnx.dataset  import ReflectDataset
from refnx.analysis import Objective, CurveFitter

from dynesty import NestedSampler
from dynesty import plotting as dyplot

from simulate    import simulate_noisy, vary_model
from information import calc_FIM

class Sampler:
    def __init__(self, objective):
        self.objective = objective
        self.ndim = len(objective.varying_parameters())
        self.sampler = NestedSampler(self.logl, self.objective.prior_transform, self.ndim)
    
    def sample(self):
        self.sampler.run_nested()
        results = self.sampler.results
        fig, _ = dyplot.cornerplot(results, color='blue', quantiles=None, show_titles=True, max_n_ticks=3, truths=np.zeros(self.ndim), truth_color='black')
        return fig
    
    def logl(self, x):
        #Update the model
        for i, parameter in enumerate(self.objective.varying_parameters()):
            parameter.value = x[i]
        return self.objective.logl()

def fisher(structure, angle, points, time, method="MCMC"):
    model, q, r, r_error, flux = simulate_noisy(structure, angle, points, time)
    vary_model(model)
    objective = Objective(model, ReflectDataset([q, r, r_error]))
    
    if method == "MCMC":
        fitter = CurveFitter(objective)
        fitter.fit('differential_evolution')
        fitter.sample(400)
        fitter.reset()
        fitter.sample(30, nthin=100)
        fig = objective.corner()
    elif method == "Nested-Sampling":
        sampler = Sampler(objective)
        fig = sampler.sample()
    else:
        return

    xi = objective.varying_parameters()
    g = calc_FIM(q, r, xi, flux, model)
    plot_ellipses(g, xi, fig)

def plot_ellipses(g, xi, fig):
    axes = np.reshape(np.array(fig.get_axes()), (len(xi), len(xi)))
    m = len(xi)
    for i in range(m):
        for j in range(m):
            if i > j:
                confidence_ellipse(g, j, i, xi[j], xi[i], axes[i,j], i==m-1, j==0)
            elif i == j:
                continue
            else:
                axes[i,j].set_visible(False)
    plt.show()

def confidence_ellipse(fisher, i, j, param1, param2, axis, show_xlabel, show_ylabel, k=1):
    g = [[fisher[i,i], fisher[i,j]], [fisher[j,i], fisher[j,j]]]

    x = []
    y = []
    for theta in np.arange(0, 2*np.pi, 0.001):
        X = np.array([np.sin(theta), np.cos(theta)])
        epsilon = k / np.sqrt(np.dot(np.dot(X, g), X.T))
        x.append(epsilon*np.sin(theta))
        y.append(epsilon*np.cos(theta))

    x = np.array(x) + param1.value
    y = np.array(y) + param2.value
    
    axis.set_xlim(param1.value*0.99995, param1.value*1.00005)
    axis.set_ylim(param2.value*0.99995, param2.value*1.00005) 
    axis.plot(x,y, color='red')
    if show_xlabel:
        axis.set_xlabel(param1.name)
    if show_ylabel:
        axis.set_ylabel(param2.name)
        
if __name__ == "__main__":
    from structures import similar_sld_sample_1, similar_sld_sample_2
    from structures import thin_layer_sample_1,  thin_layer_sample_2
    from structures import easy_sample, many_param_sample
    
    structure = easy_sample()
    angle      = 0.7
    time       = 100000
    points     = 200
    
    fisher(*structure, angle, points, time, method="Nested-Sampling")
    fisher(*structure, angle, points, time, method="MCMC")
