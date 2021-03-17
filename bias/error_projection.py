import matplotlib.pyplot as plt
import numpy as np
import os

from refnx.dataset import ReflectDataset
from refnx.reflect import ReflectModel
from refnx.analysis import Objective, CurveFitter

from structures import QCS_sample
from simulate import simulate
from utils import vary_structure

def plot_simulated_projections(structure, angle_times, time_factors):
    errors = []
    for i, factor in enumerate(time_factors, 1):
        new_angle_times = {angle: (angle_times[angle][0], angle_times[angle][1]*factor) 
                           for angle in angle_times}
        
        objective = Objective(*simulate(vary_structure(structure(), random=False, bound_size=0.25), new_angle_times))
        CurveFitter(objective).fit('differential_evolution', verbose=False)

        errors.append([param.stderr for param in objective.varying_parameters()])

        print("{0}/{1}".format(i, len(reduction_range)))

    fig = plt.figure(figsize=[7,5], dpi=600)
    ax = fig.add_subplot(111)

    errors = np.asarray(errors)
    for i, param in enumerate(objective.varying_parameters()):
        ax.plot(time_factors, errors[:,i], label=param.name, linestyle="", marker="x", zorder=2)

        times = np.arange(time_factors[0], time_factors[-1], 0.01)
        ax.plot(times, errors[0][i]/np.sqrt(times), color='black', lw=0.8, zorder=1)

    ax.set_xlabel('Time')
    ax.set_ylabel('Parameter Uncertainty')
    ax.legend()
    
    save_path = "./results/error_projection"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    fig.savefig(save_path+"/{}.png".format(structure.__name__))

def plot_measured_projections():
    datasets = [np.loadtxt('./data/{}.dat'.format(i+1), delimiter=',') for i in range(10)]

    errors = []
    for i in range(1, 11):
        points = 96*i
        data = np.zeros((points, 3))
        q, r, r_error = [], [], []
        for sliced in datasets[:i]:
            q += sliced[:,0].tolist()
            r += sliced[:,1].tolist()
            r_error += sliced[:,2].tolist()

        data[:,0] = q
        data[:,1] = r
        data[:,2] = r_error

        data = data[(data != 0).all(1)]  #Remove points of zero reflectivity.
        data = data[data[:,0].argsort()] #Sort by Q

        refdata = ReflectDataset([data[:,0], data[:,1], data[:,2]])
        refdata.scale(np.max(refdata.y))

        model = ReflectModel(vary_structure(QCS_sample()), dq=2.5, scale=1, bkg=8e-7)
        objective = Objective(model, refdata)
        CurveFitter(objective).fit('differential_evolution', verbose=False)

        errors.append([param.stderr for param in objective.varying_parameters()])

    fig = plt.figure(figsize=[7,5], dpi=600)
    ax = fig.add_subplot(111)

    errors = np.asarray(errors)
    for i, param in enumerate(objective.varying_parameters()):
        ax.plot(np.arange(1, len(errors[:,i])+1), errors[:,i], label=param.name, linestyle="", marker="x", zorder=2)

        times = np.arange(1, len(errors[:,i])+1, 0.01)
        ax.plot(times, errors[0][i]/np.sqrt(times), color='black', lw=0.8, zorder=1)

    ax.set_xlabel('Time')
    ax.set_ylabel('Parameter Uncertainty')
    ax.legend()
    
    save_path = "./results/error_projection"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fig.savefig(save_path+"/QCS_measured.png")

if __name__ == "__main__":
    from structures import similar_sld_sample_1, similar_sld_sample_2
    from structures import thin_layer_sample_1, thin_layer_sample_2
    from structures import easy_sample, many_param_sample
    
    plot_measured_projections()
    
    structure = easy_sample
    initial_angle_times = {0.7: (70, 5),
                           2.0: (70, 20)}
    reduction_range = np.arange(1, 10, 0.25)
    plot_simulated_projections(structure, initial_angle_times, reduction_range)
