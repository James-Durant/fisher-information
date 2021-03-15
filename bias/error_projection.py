import matplotlib.pyplot as plt
import numpy as np

from refnx.dataset import ReflectDataset
from refnx.reflect import SLD, ReflectModel
from refnx.analysis import Objective, CurveFitter

from structures import *

def QCS_sample():
    air       = SLD(0,      name="Air")
    layer1    = SLD(1.795,  name="Layer 1 - Si")(thick=790.7,   rough=24.5)
    layer2    = SLD(6.385,  name="Layer 2 - Cu")(thick=297.9,   rough=3.5)
    substrate = SLD(3.354,  name="Substrate - Quartz")(thick=0, rough=12.9)
    structure = air | layer1 | layer2 | substrate
    return vary_structure(structure)

def plot_projections(structure, initial_angle_times, reduction_range):
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

        model = ReflectModel(QCS_sample(), dq=2.5, scale=1, bkg=8e-7)
        objective = Objective(model, refdata)
        CurveFitter(objective).fit('differential_evolution')

        if i == 1:
            inital_errors = [param.stderr for param in objective.varying_parameters()]

        errors.append([param.stderr for param in objective.varying_parameters()])

    fig = plt.figure(figsize=[7,5], dpi=600)
    ax = fig.add_subplot(111)

    errors = np.asarray(errors)
    for i, param in enumerate(objective.varying_parameters()):
        ax.plot(np.arange(1, len(errors[:,i])+1), errors[:,i], label=param.name, linestyle="", marker="x", zorder=2)

        times = np.arange(1, len(errors[:,i])+1, 0.01)
        ax.plot(times, inital_errors[i]/np.sqrt(times), color='black', lw=0.8, zorder=1)

    ax.set_xlabel('Time')
    ax.set_ylabel('Parameter Uncertainty')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    structure = easy_sample
    angle_times = {0.7: (70, 5),
                   2.0: (70, 20)}
    reduction_range = np.arange(1, 10, 0.5)
    plot_projections(structure, angle_times, reduction_range)
