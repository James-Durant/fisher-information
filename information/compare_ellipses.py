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
    """The Sampler class contains the code nested sampling.

    Attributes:
        objective (refnx.analysis.Objective): the objective to fit.
        ndim (int): the dimensionalilty of the problem.
        sampler (dynesty.NestedSampler): a dynesty static nested sampler.

    """
    def __init__(self, objective):
        self.objective = objective
        self.ndim = len(objective.varying_parameters())
        self.sampler = NestedSampler(self.logl, self.objective.prior_transform, self.ndim)

    def sample(self):
        """Fits the given objective by sampling.

        Returns:
            matplotlib.pyplot.figure: corner plot figure from the sampling routine.

        """
        self.sampler.run_nested()
        results = self.sampler.results
        fig, _ = dyplot.cornerplot(results, color='blue', quantiles=None, show_titles=True, max_n_ticks=3, truths=np.zeros(self.ndim), truth_color='black')
        return fig

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

def fisher(structure, angle, points, time, method="MCMC"):
    """Fits a model for a given `structure` using a given `method`, calculates
       the Fisher information and plots the Fisher confidence ellipses against
       the `method`'s corner plot.

    Args:
        structure (refnx.reflect.Structure): the structure to simulate the experiment for.
        angle (float): measurement angle to use in the experiment simulation.
        points (int): number of points to obtain from binning in the experiment simulation.
        time (int): how long to measure for in the experiment simulation.
        method (string): either 'MCMC' or 'Nested-Sampling'

    """
    #Simulate an experiment using the given angle, number of points and time.
    model, q, r, r_error, flux = simulate_noisy(structure, angle, points, time)
    vary_model(model) #Vary the SLD and thickness of each layer and set them to random values.
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

    #Calculate the Fisher information matrix, g, and plot the confidence ellipses.
    xi = objective.varying_parameters()
    g = calc_FIM(q, r, xi, flux, model)
    plot_ellipses(g, xi, fig)

def plot_ellipses(g, xi, fig):
    """Plots the Fisher confidence ellipses against the corner plot of either an
       MCMC or nested sampling result.

    Args:
        g (numpy.ndarray): the Fisher information matrix.
        xi (list): a list of refnx Parameter objects representing each varying parameter.
        fig (matplotlib.pyplot.figure): the corner plot of either MCMC or nested sampling.

    """
    #Get the existing axes from the corner plot.
    axes = np.reshape(np.array(fig.get_axes()), (len(xi), len(xi)))
    m = len(xi)
    for i in range(m):
        for j in range(m):
            if i > j: #Plot the confidence ellipise on the plots below the diagonal.
                confidence_ellipse(g, j, i, xi[j], xi[i], axes[i,j], i==m-1, j==0)
            elif i == j:
                continue #Leave diagonal plots as they are.
            else:
                axes[i,j].set_visible(False) #Remove all other plots.
    plt.show()

def confidence_ellipse(fisher, i, j, param1, param2, axis, show_xlabel, show_ylabel, k=1):
    """Plots the confidence ellipse between `param1` and `param2`.

    Args:
        fisher (numpy.ndarray): the Fisher information matrix (FIM).
        i (int): index of `param1` in the FIM.
        j (int): index of `param2` in the FIM.
        param1 (refnx.analysis.Parameter): the first parameter corresponding to `i`.
        param2 (refnx.analysis.Parameter): the second parameter corresponding to `j`.
        axis (matplotlib.pyplot.axis): the subplot of the corner plot to plot on.
        show_xlabel (Boolean): whether to display the x-axis label.
        show_ylabel (Boolean): whether to display the y-axis label.
        k (int): the number of standard deviations.

    """
    #Retrieve the elements of the FIM for the given parameters.
    g = [[fisher[i,i], fisher[i,j]], [fisher[j,i], fisher[j,j]]]

    #Calculate the values of the confidence ellipse.
    x, y = [], []
    for theta in np.arange(0, 2*np.pi, 0.001):
        X = np.array([np.sin(theta), np.cos(theta)])
        epsilon = k / np.sqrt(np.dot(np.dot(X, g), X.T))
        x.append(epsilon*np.sin(theta))
        y.append(epsilon*np.cos(theta))

    #Move the confidence ellipse to be centred on the parameter estimates
    x = np.array(x) + param1.value
    y = np.array(y) + param2.value

    #Adjust the x and y axes so the corner plot contours and new ellipse can be seen.
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