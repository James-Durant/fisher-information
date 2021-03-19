import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import no_protein

from simulate import simulate
from utils import calc_FIM

from refnx.dataset import ReflectDataset
from refnx.analysis import Objective, GlobalObjective, CurveFitter

def plot_objective(objective, show_fit=True):
    """Plots the fit of a given `objective` against the objective's data.

    Args:
        objective (refnx.analysis.Objective): the objective to plot.
        show_fit (Boolean): whether to display the objective fit or not.

    """
    fig = plt.figure(figsize=[9,7], dpi=600)
    ax = fig.add_subplot(111)

    #Add the data.
    q = objective.data.x
    r, r_error, r_model = objective._data_transform(model=objective.generative())
    ax.errorbar(q, r, r_error, color="black", marker="o", ms=3, lw=0, elinewidth=1, capsize=1.5)

    if show_fit: #Add the prediction/fit.
        ax.plot(q, r_model, color="red", zorder=20)

    ax.set_yscale('log')
    ax.set_xlim(0, 0.3)
    ax.set_ylim(1e-7, 2)
    return fig

def compare_ellipses(angle_times):
    info = {
        "Run Number": [
            "19584_85",
            "19598_99",
            "19602_03",
            "19604_05",
            "19606_07",
            "19608_09",
        ],
        "DPPC Contrast": ["D", "H", "D", "H", "D", "H"],
        "Water Contrast": ["D2O", "D2O", "CMSi", "CMSi", "H2O", "H2O"],
    }
    
    experiment = pd.DataFrame(info)
    
    experiment["Experimental Data"] = no_protein.get_data(experiment)
    experiment["Structures"] = no_protein.get_structures(experiment)
    experiment["Models"] = no_protein.get_models(experiment)
    experiment["Objectives"] = no_protein.get_objective(experiment)
    experiment = no_protein.set_constraints(experiment)
    
    #Losing parameters somewhere 
    g_objective = GlobalObjective([i for i in experiment["Objectives"]])
    
    qs, counts, models, objectives = [], [], [], []
    for objective in g_objective.objectives:
        model, data = simulate(objective.model.structure, angle_times, include_counts=True,
                               dq=objective.model.dq)
        
        qs.append(data[:,0])
        counts.append(data[:,3])
        models.append(model)
        objectives.append(Objective(model, ReflectDataset([data[:,0], data[:,1], data[:,2]])))
    
    g_objective = GlobalObjective(objectives)
    fitter = CurveFitter(g_objective)
    fitter.fit('differential_evolution')
    fitter.sample(200)
    fitter.reset()
    fitter.sample(200, 10)
    fig = g_objective.corner()
    
    xi = g_objective.varying_parameters()
    g = calc_FIM(qs, xi, np.concatenate(counts), models)    
    
    plot_ellipses(g, xi, fig)
    fig.savefig("./results/asymmetric_bilayer.png")

def plot_ellipses(g, xi, fig):
    """Plots the FIM confidence ellipses against the corner plot from either
       MCMC or nested sampling.

    Args:
        g (numpy.ndarray): the Fisher information metric matrix.
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

def confidence_ellipse(g, i, j, param1, param2, axis, show_xlabel, show_ylabel):
    """Plots the confidence ellipse between `param1` and `param2`.

    Args:
        g (numpy.ndarray): the Fisher information metric matrix.
        i (int): index of `param1` in the FIM.
        j (int): index of `param2` in the FIM.
        param1 (refnx.analysis.Parameter): the first parameter corresponding to `i`.
        param2 (refnx.analysis.Parameter): the second parameter corresponding to `j`.
        axis (matplotlib.pyplot.axis): the subplot of the corner plot to plot on.
        show_xlabel (Boolean): whether to display the x-axis label.
        show_ylabel (Boolean): whether to display the y-axis label.

    """
    #Retrieve the elements of the FIM for the given parameters.
    g_params = [[g[i,i], g[i,j]], [g[j,i], g[j,j]]]

    #Iterative over k: the number of standard deviations.
    for k in [1,2,3]:
        #Calculate the values of the confidence ellipse.
        x, y = [], []
        for theta in np.arange(0, 2*np.pi, 0.001):
            X = np.array([np.sin(theta), np.cos(theta)])
            epsilon = k / np.sqrt(np.dot(np.dot(X, g_params), X.T))
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
    angle_times = {0.7: (70, 5), #Angle: (Points, Time)
                   2.0: (70, 20)}
    
    compare_ellipses(angle_times)
