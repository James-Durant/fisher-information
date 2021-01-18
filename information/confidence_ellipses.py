import numpy as np
import os

from refnx.dataset  import ReflectDataset
from refnx.analysis import Objective

from simulation.simulate import simulate_single_contrast, vary_model, plot_objective
from utils import Sampler, calc_FIM

def compare_ellipses(structure, angle_times, save_path):
    """Fits a model for a given `structure` using MCMC and nested sampling,
       calculates the Fisher information and plots the Fisher confidence ellipses
       against the corner plots.

    Args:
        structure (refnx.reflect.Structure): the structure to simulate the experiment for.
        angle_times (dict): dictionary of times and number of points for each measurement angle.
        save_path (string): path to directory for saving data and figures.

    """
    #Simulate an experiment using the given angles, number of points and times.
    model, data = simulate_single_contrast(structure, angle_times, save_path=save_path)

    #Retrieve data from simulation dataset and create objective to fit.
    q, r, r_error, flux = data[:,0], data[:,1], data[:,2], data[:,3]
    objective = Objective(model, ReflectDataset([q, r, r_error]))
    vary_model(model)

    #Perform MCMC sampling and get the resulting corner plot.
    sampler = Sampler(objective)
    fig1, _ = sampler.sample_MCMC(verbose=True)
    fig2    = plot_objective(objective)
    xi = objective.varying_parameters()
    g  = calc_FIM(q, xi, flux, model) #Calculate the Fisher information matrix
    plot_ellipses(g, xi, fig1) #Plot confidence ellipses on corner plots.
    fig1.savefig(save_path+"/confidence_ellipses_MCMC.png", dpi=600)
    fig2.savefig(save_path+"/fit_MCMC.png", dpi=600)

    #Perform nested sampling and overlay confidence ellipses on corner plot.
    fig3, _ = sampler.sample_nested(verbose=True)
    fig4    = plot_objective(objective)
    xi = objective.varying_parameters()
    g  = calc_FIM(q, xi, flux, model)
    plot_ellipses(g, xi, fig3)
    fig3.savefig(save_path+"/confidence_ellipses_nested.png", dpi=600)
    fig4.savefig(save_path+"/fit_nested.png", dpi=600)

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
            if i > j: #Plot the confidence ellipse on the plots below the diagonal.
                confidence_ellipse(g, j, i, xi[j], xi[i], axes[i,j], i==m-1, j==0)
            elif i == j:
                continue #Leave diagonal plots as they are.
            else:
                axes[i,j].set_visible(False) #Remove all other plots.

    axes[m-1,m-1].set_xlabel(xi[i].name)

def confidence_ellipse(fisher, i, j, param1, param2, axis, show_xlabel, show_ylabel):
    """Plots the confidence ellipse between `param1` and `param2`.

    Args:
        fisher (numpy.ndarray): the Fisher information matrix.
        i (int): index of `param1` in the FIM.
        j (int): index of `param2` in the FIM.
        param1 (refnx.analysis.Parameter): the first parameter corresponding to `i`.
        param2 (refnx.analysis.Parameter): the second parameter corresponding to `j`.
        axis (matplotlib.pyplot.axis): the subplot of the corner plot to plot on.
        show_xlabel (Boolean): whether to display the x-axis label.
        show_ylabel (Boolean): whether to display the y-axis label.

    """
    #Retrieve the elements of the FIM for the given parameters.
    g = [[fisher[i,i], fisher[i,j]], [fisher[j,i], fisher[j,j]]]

    #Iterative over k: the number of standard deviations.
    for k in [1,2,3]:
        #Calculate the values of the confidence ellipse.
        x, y = [], []
        for theta in np.arange(0, 2*np.pi, 0.001):
            X = np.array([np.sin(theta), np.cos(theta)])
            epsilon = k / np.sqrt(np.dot(np.dot(X, g), X.T))
            x.append(epsilon*np.sin(theta))
            y.append(epsilon*np.cos(theta))

        #Move the confidence ellipse to be centred on the parameter estimates.
        x = np.array(x) + param1.value
        y = np.array(y) + param2.value
        axis.plot(x,y, color='r')

    if show_xlabel:
        axis.set_xlabel(param1.name)
    if show_ylabel:
        axis.set_ylabel(param2.name)

if __name__ == "__main__":
    from simulation.structures import similar_sld_sample_1, similar_sld_sample_2
    from simulation.structures import thin_layer_sample_1,  thin_layer_sample_2
    from simulation.structures import easy_sample, many_param_sample

    structure   = easy_sample #Choose structure here.
    angle_times = {0.7: (70, 5), #Angle: (Points, Time)
                   2.0: (70, 20)}

    save_path = "./results/"+structure.__name__
    if not os.path.exists(save_path): #Create directory if not present.
        os.makedirs(save_path)

    #Overlay the Fisher confidence ellipses on the MCMC and nested sampling corner plots.
    compare_ellipses(structure(), angle_times, save_path)
